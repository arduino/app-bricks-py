# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The runtime that runs a listed model and the publisher that distributes it.

Both are reported by the listing as ``runtime`` and ``model_publisher``, and both are
derived from what the handler downloads rather than declared for every entry:

- ``runtime`` follows the handler alone: every AI Hub model runs on QNN, every Edge
  Impulse model is an ``.eim`` built on the Edge Impulse SDK, whatever the target it was
  built for, and every Hugging Face model is a GGUF served by llama.cpp.
- ``model_publisher`` is who publishes the downloaded artifact, not who authored the
  underlying model: Qualcomm AI Hub for AI Hub, Edge Impulse for Edge Impulse, and the
  repository owner for Hugging Face ("unsloth" for "unsloth/gemma-3-1b-it-GGUF"), kept
  in the owner's own case. A models-list.yaml entry overrides it with
  ``metadata.model_publisher`` — an Edge Impulse project built on an AI Hub model is
  published as ``qualcomm-ai-hub``. A canonical Hugging Face repository has no owner,
  and its publisher is None.
"""

from urllib.parse import unquote, urlsplit

RUNTIME_QNN = "qnn"
RUNTIME_EDGE_IMPULSE_SDK = "edge-impulse-sdk"
RUNTIME_LLAMACPP = "llamacpp"

PUBLISHER_QUALCOMM_AI_HUB = "qualcomm-ai-hub"
PUBLISHER_EDGE_IMPULSE = "edge-impulse"

# "llamacpp" is the handler the listing reports for a GGUF found on disk.
HANDLER_RUNTIMES = {
    "ai-hub-handler": RUNTIME_QNN,
    "ei-handler": RUNTIME_EDGE_IMPULSE_SDK,
    "hf-handler": RUNTIME_LLAMACPP,
    "llamacpp": RUNTIME_LLAMACPP,
}

HANDLER_PUBLISHERS = {
    "ai-hub-handler": PUBLISHER_QUALCOMM_AI_HUB,
    "ei-handler": PUBLISHER_EDGE_IMPULSE,
}

_HF_NETLOC = "huggingface.co"
_HF_FILE_MARKERS = ("resolve", "blob")


def model_runtime(handler):
    """The runtime that runs a model of *handler*, or None for an unknown handler."""
    return HANDLER_RUNTIMES.get(handler)


def hf_repo_id(model_url):
    """The Hugging Face repository id *model_url* names, or None when it names none.

    Reads both syntaxes hf_downloader accepts: a huggingface.co URL (of a file, or of
    the repository itself), and a compact key ``[<model_type>:]<repo_id>[:<quantization>
    [:<mmproj_quantization>]]``. Nothing is validated — the downloader already did that
    before anything reached the disk.
    """
    model_url = (model_url or "").strip()
    if not model_url:
        return None
    if model_url.startswith(("http://", "https://")):
        parts = urlsplit(model_url)
        if parts.netloc.lower() != _HF_NETLOC:
            return None
        segments = [unquote(segment) for segment in parts.path.split("/") if segment]
        marker = next((i for i, segment in enumerate(segments) if segment in _HF_FILE_MARKERS), len(segments))
        return "/".join(segments[:marker]) if 1 <= marker <= 2 else None
    fields = model_url.split(":")
    # One or two fields start with the repository; three or four start with model_type.
    repo_id = fields[0] if len(fields) <= 2 else fields[1]
    return repo_id or None


def hf_publisher(model_url="", model_directory=""):
    """The owner of the Hugging Face repository a model was downloaded from, or None.

    *model_url* is authoritative; *model_directory* — the download directory under the
    llamacpp tree, which is the repository id unless models-list.yaml says otherwise —
    covers a model whose URL is unknown. Either way the id must be ``<owner>/<repo>``:
    a canonical repository has no owner.
    """
    repo_id = hf_repo_id(model_url)
    if repo_id is None:
        repo_id = (model_directory or "").strip("/")
    owner, sep, _repo = repo_id.partition("/")
    return owner if sep and owner else None


def model_publisher(handler, metadata=None, model_url="", model_directory=""):
    """The publisher of a model of *handler*; see the module docstring.

    Args:
        handler: The handler of the entry ("ai-hub-handler", "ei-handler",
            "hf-handler", or "llamacpp" for a GGUF found on disk).
        metadata: The entry's models-list.yaml ``metadata``; its ``model_publisher``
            wins over anything derived.
        model_url: For Hugging Face, the URL or compact key the model was downloaded from.
        model_directory: For Hugging Face, the model's directory under the llamacpp tree.
    """
    declared = (metadata or {}).get("model_publisher")
    if declared:
        return str(declared)
    if HANDLER_RUNTIMES.get(handler) == RUNTIME_LLAMACPP:
        return hf_publisher(model_url, model_directory)
    return HANDLER_PUBLISHERS.get(handler)
