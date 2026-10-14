# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Thin ONNX Runtime wrapper.

Hides the three things that differ between the exported EasyOCR .onnx variants:
  * execution provider - QNN/HTP when the NPU is reachable, CPU otherwise
    (see `utils.onnx_ep`).
  * tensor layout - the ai-hub-models ONNX exports are NCHW, unlike the TFLite ones
    which get rewritten to NHWC. The pipeline speaks NHWC, so we transpose here.
  * quantization - the float export (the one shipped) takes and returns float32 and
    needs none of this; the w8a8 export takes and returns real uint8 tensors, and ONNX
    keeps no quantization parameters on the graph inputs/outputs. They live in the
    `metadata.json` that ai-hub ships next to the models, so that file is read here
    whenever the graph I/O is integer.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort

from utils.onnx_ep import QNN_EP_NAME, build_session, summarize_profile

# ORT advertises tensor element types as strings.
ORT_TYPE_TO_NUMPY: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
}


def dequantize(tensor: np.ndarray, zero_point: float, scale: float) -> np.ndarray:
    """Map an integer tensor back to float using its quantization parameters."""
    return (tensor.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def quantize(array: np.ndarray, zero_point: float, scale: float, dtype: np.dtype) -> np.ndarray:
    """Map a float array onto an integer tensor using its quantization parameters."""
    info = np.iinfo(dtype)
    quantized = np.round(array / np.float32(scale)) + np.float32(zero_point)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _detect_layout(shape: tuple[int, ...]) -> str:
    """
    Guess whether a 4D input tensor is NHWC or NCHW.

    The channel axis is the only one that can hold 1 (grey) or 3 (RGB) for these two
    models, and the spatial axes are always much larger, so the ambiguity resolves.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D input tensor, got shape {shape}")
    channels_last = shape[3] in (1, 3)
    channels_first = shape[1] in (1, 3)
    if channels_last and not channels_first:
        return "NHWC"
    if channels_first and not channels_last:
        return "NCHW"
    raise ValueError(f"Cannot infer tensor layout from input shape {shape}")


def _load_quantization_metadata(model_path: str) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Read the per-tensor (scale, zero_point) pairs ai-hub-models writes next to the graph.

    Returns
    -------
    params : dict
        ``{"inputs": {name: (scale, zero_point)}, "outputs": {...}}``; empty when there is
        no metadata.json, which is the normal case for a float export.
    """
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "metadata.json")
    empty: dict[str, dict[str, tuple[float, float]]] = {"inputs": {}, "outputs": {}}
    if not os.path.isfile(metadata_path):
        return empty

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    entry = metadata.get("model_files", {}).get(os.path.basename(model_path))
    if entry is None:
        return empty

    params: dict[str, dict[str, tuple[float, float]]] = {"inputs": {}, "outputs": {}}
    for section in ("inputs", "outputs"):
        for name, spec in entry.get(section, {}).items():
            quantization = spec.get("quantization_parameters")
            if quantization:
                params[section][name] = (
                    float(quantization["scale"]),
                    float(quantization["zero_point"]),
                )
    return params


class ONNXModel:
    """A single-input ONNX model that consumes NHWC float arrays."""

    def __init__(
        self,
        model_path: str,
        backend: str | None = None,
        intra_op_threads: int | None = None,
        verify_placement: bool | None = None,
    ) -> None:
        self.model_path = model_path

        # Asking for the NPU explicitly means the caller wants to know it got it. Creating
        # a QNN session proves nothing: ORT builds one even when QNN's GetCapability fails
        # and the EP claims zero nodes, and get_providers() still lists QNN afterwards.
        if verify_placement is None:
            verify_placement = (backend or "").lower() == "qnn" or os.environ.get("EASYOCR_QNN_VERIFY", "0") == "1"
        self.verify_placement = verify_placement

        self.session, attached_provider = build_session(
            model_path,
            backend=backend,
            intra_op_threads=intra_op_threads,
            profile=verify_placement,
        )
        self.execution_provider = attached_provider
        self.placement: dict[str, tuple[int, float]] | None = None
        self.run_options = self._build_run_options()

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{model_path} has {len(inputs)} inputs, this wrapper handles exactly one")
        model_input = inputs[0]

        self.input_name = model_input.name
        self.input_dtype = self._numpy_dtype(model_input.type, model_input.name)

        input_shape = self._resolve_shape(model_input.shape)
        self.layout = _detect_layout(input_shape)
        if self.layout == "NHWC":
            _, self.height, self.width, self.channels = input_shape
        else:
            _, self.channels, self.height, self.width = input_shape

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]
        self.output_dtypes = [self._numpy_dtype(output.type, output.name) for output in outputs]
        self.output_shapes = [tuple(output.shape) for output in outputs]

        quantization = _load_quantization_metadata(model_path)
        self.input_quantization = quantization["inputs"].get(self.input_name)
        self.output_quantization = [quantization["outputs"].get(name) for name in self.output_names]

        if self.input_dtype.kind in "iu" and self.input_quantization is None:
            raise RuntimeError(
                f"{model_path} takes a {self.input_dtype} input but no quantization parameters "
                "were found. ai-hub-models ships them in the metadata.json next to the model - "
                "keep that file next to the .onnx."
            )

        if verify_placement:
            self.placement = self._measure_placement()
            self.execution_provider = self._provider_from_placement(self.placement, attached_provider)
            if (backend or "").lower() == "qnn" and QNN_EP_NAME not in self.placement:
                raise RuntimeError(
                    f"{model_path}: the QNN EP was attached but executed no nodes - the whole "
                    f"graph ran on {self.execution_provider}. The ORT log above carries the QNN "
                    "error (a failed GetCapability / SetupBackend usually means a bad "
                    "ADSP_LIBRARY_PATH, a QAIRT/skel version mismatch, or an htp_arch the "
                    "device does not have). Use --ep auto to run on the CPU meanwhile."
                )

    def _measure_placement(self) -> dict[str, tuple[int, float]]:
        """
        Run one inference and report which providers really executed nodes.

        Profiling stops at `end_profiling()`, so the cost is paid once at startup and the
        session runs unencumbered afterwards.
        """
        height, width = (self.height, self.width)
        probe = np.zeros((1, height, width, self.channels), dtype=np.float32)
        self(probe)
        profile_path = self.session.end_profiling()
        try:
            return summarize_profile(profile_path)
        finally:
            try:
                os.remove(profile_path)
            except OSError:
                pass

    @staticmethod
    def _provider_from_placement(placement: dict[str, tuple[int, float]], fallback: str) -> str:
        """The provider that actually did the work, by execution time."""
        if not placement:
            return fallback
        return max(placement.items(), key=lambda item: item[1][1])[0]

    @staticmethod
    def _numpy_dtype(ort_type: str, tensor_name: str) -> np.dtype:
        try:
            return ORT_TYPE_TO_NUMPY[ort_type]
        except KeyError:
            raise ValueError(f"Unsupported element type {ort_type} on tensor {tensor_name!r}") from None

    @staticmethod
    def _resolve_shape(shape: list) -> tuple[int, ...]:
        """
        Turn an ORT input shape into concrete ints.

        Symbolic dimensions come back as strings (or None). Only the batch axis may be
        symbolic - the QNN HTP backend rejects dynamic shapes outright, so a dynamic
        spatial or channel axis means the export has to be redone with fixed dimensions.
        """
        resolved: list[int] = []
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, int):
                resolved.append(dimension)
            elif axis == 0:
                resolved.append(1)  # symbolic batch, this pipeline always feeds 1
            else:
                raise ValueError(
                    f"{shape} has a dynamic axis {axis} ({dimension!r}). The QNN HTP backend "
                    "requires static shapes - re-export the model with fixed input dimensions."
                )
        return tuple(resolved)

    @staticmethod
    def _build_run_options() -> ort.RunOptions | None:
        """Per-run QNN knobs. Only built when something actually needs setting."""
        latency = os.environ.get("EASYOCR_QNN_RPC_LATENCY")
        if not latency:
            return None
        run_options = ort.RunOptions()
        run_options.add_run_config_entry("qnn.rpc_control_latency", latency)
        return run_options

    @property
    def image_shape(self) -> tuple[int, int]:
        """Network input resolution as (height, width)."""
        return (self.height, self.width)

    def __call__(self, nhwc_input: np.ndarray) -> list[np.ndarray]:
        """
        Run the model.

        Parameters
        ----------
        nhwc_input
            [N, H, W, C] float32 array already scaled to the range the network expects.

        Returns
        -------
        outputs : list[np.ndarray]
            One dequantized float32 array per model output, in graph output order.
        """
        tensor = nhwc_input if self.layout == "NHWC" else nhwc_input.transpose(0, 3, 1, 2)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        if self.input_dtype != np.float32:
            if self.input_quantization is not None:
                scale, zero_point = self.input_quantization
                tensor = quantize(tensor, zero_point, scale, self.input_dtype)
            else:
                tensor = tensor.astype(self.input_dtype)

        raw_outputs = self.session.run(self.output_names, {self.input_name: tensor}, self.run_options)

        outputs = []
        for raw, quantization in zip(raw_outputs, self.output_quantization):
            if raw.dtype.kind in "iu" and quantization is not None:
                scale, zero_point = quantization
                outputs.append(dequantize(raw, zero_point, scale))
            else:
                outputs.append(raw.astype(np.float32))
        return outputs
