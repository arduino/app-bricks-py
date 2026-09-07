# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The cache the enrollment leaves in the poses folder between two starts."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from arduino.app_utils import Logger

from ..pose_classifier import EMBEDDING_SIZE

logger = Logger("PoseEstimation")

CACHE_FILE = Path(".cache") / "enrollment.npz"


class EnrollmentCache:
    """What the enrollment keeps from one start to the next: what reading each photo gave (its embedding,
    or the reason it was discarded), keyed by path, size and modification time and valid for one
    classifier asset, brick version and pair of reading settings, plus the pose declaration the last
    reports were written for."""

    def __init__(self, path: Path, asset_path: Path, min_score: float, out_of_frame_tolerance: float) -> None:
        self._path = path
        self.stamp = {
            "asset": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "brick": version("arduino_app_bricks"),
            "min_score": min_score,
            "out_of_frame_tolerance": out_of_frame_tolerance,
        }
        self.embeddings: dict[str, np.ndarray] = {}
        self.reasons: dict[str, str] = {}
        self.declaration = ""
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data["meta_json"]))
                    if {key: meta.get(key) for key in self.stamp} == self.stamp:
                        self.embeddings = dict(zip(data["keys"].astype(str), data["embeddings"], strict=True))
                        self.reasons = meta.get("reasons", {})
                        self.declaration = meta.get("declaration", "")
            except Exception as e:
                logger.warning(f"ignoring the unreadable embedding cache {path}: {e}")

    @staticmethod
    def key(root: Path, path: Path) -> str:
        stat = path.stat()
        return f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}"

    def store(self, key: str, embedding: np.ndarray | None, reason: str | None) -> None:
        self.embeddings[key] = embedding if embedding is not None else np.full(EMBEDDING_SIZE, np.nan, np.float32)
        if reason is not None:
            self.reasons[key] = reason

    def save(self, live_keys: list[str]) -> None:
        """Write the cache with the photos still in the folders, replacing the file in one step."""
        live = [key for key in live_keys if key in self.embeddings]
        meta = {**self.stamp, "declaration": self.declaration, "reasons": {key: self.reasons[key] for key in live if key in self.reasons}}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_suffix(".partial")
        with open(partial, "wb") as file:
            np.savez(
                file,
                keys=np.asarray(live, dtype=str),
                embeddings=np.vstack([self.embeddings[key].reshape(1, -1) for key in live]) if live else np.empty((0, EMBEDDING_SIZE), np.float32),
                meta_json=np.asarray(json.dumps(meta)),
            )
        partial.replace(self._path)
