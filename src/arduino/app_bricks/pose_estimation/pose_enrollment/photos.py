# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The photo folders of the custom poses: reading them through the model runner and the buckets the
enrollment measures."""

import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from websockets.sync.client import connect

from arduino.app_utils.image.adjustments import compress_to_jpeg

from ..pose_classifier import EMBEDDING_SIZE, embed_person
from ..pose_types import Person, parse_people
from .cache import EnrollmentCache
from .measure import OTHER, Bucket

FOLDER_NAME = re.compile(r"[a-z][a-z0-9_]*")
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")
CONNECT_TIMEOUT_SEC = 60.0
ANSWER_TIMEOUT_SEC = 15.0

DISCARD_REASONS = {
    "anchors": "skeleton incomplete",
    "missing": "skeleton incomplete",
    "torso": "skeleton incomplete",
    "out_of_frame": "person partly out of frame",
}


class PersonReader:
    """Reads still images through the runner's WebSocket channels, one at a time.

    Results are read from the output socket first and frames sent on the input socket second,
    so every sent frame is matched to the next result. The runner keeps a person-tracking
    crop from one frame to the next, so a black frame clears it before the first photo, and a
    photo is sent twice (the second answer is the runner's own cropped pass) and followed by
    another black frame.
    """

    def __init__(self, send_url: str, recv_url: str, config: dict) -> None:
        self._send_url, self._recv_url, self._config = send_url, recv_url, config
        self._send = self._recv = None
        self._cleared = False

    def __enter__(self) -> "PersonReader":
        deadline = time.monotonic() + CONNECT_TIMEOUT_SEC
        while True:
            try:
                self._recv = connect(self._recv_url, open_timeout=5)
                self._send = connect(self._send_url, open_timeout=5)
                break
            except OSError as error:
                self.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"the pose model runner did not answer within {CONNECT_TIMEOUT_SEC:.0f}s: {error}") from error
                time.sleep(1.0)
        self._send.send(json.dumps({"config": self._config}))
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for socket in (self._send, self._recv):
            if socket is not None:
                socket.close()
        self._send = self._recv = None

    def _infer(self, jpeg: bytes) -> dict:
        self._send.send(json.dumps({"frame": base64.b64encode(jpeg).decode("utf-8")}))
        answer = json.loads(self._recv.recv(timeout=ANSWER_TIMEOUT_SEC))
        return answer.get("metadata", {})

    def people(self, image: np.ndarray, min_score: float) -> list[Person]:
        """Detect the people of one BGR image with the runner's two-pass reading."""
        jpeg = compress_to_jpeg(image)
        if jpeg is None:
            raise RuntimeError("the photo could not be encoded as JPEG")
        black = compress_to_jpeg(np.zeros_like(image)).tobytes()
        if not self._cleared:
            self._infer(black)
            self._cleared = True
        self._infer(jpeg.tobytes())
        metadata = self._infer(jpeg.tobytes())
        self._infer(black)
        return parse_people(metadata, min_score)


def custom_folders(root: Path, builtin_names: tuple[str, ...]) -> tuple[str, ...]:
    """The pose folders under root, refusing a folder that shadows a built-in pose or is not a valid pose name."""
    if not root.is_dir():
        return ()
    names = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith(".") or folder.name == OTHER:
            continue
        if folder.name in builtin_names:
            raise ValueError(f"pose folder {folder} carries a built-in pose name: rename it, built-in poses cannot be replaced")
        if not FOLDER_NAME.fullmatch(folder.name):
            raise ValueError(f"pose folder {folder} is not a valid pose name: use lowercase letters, digits and underscores, starting with a letter")
        names.append(folder.name)
    return tuple(names)


def photos(folder: Path) -> list[Path]:
    """The photos of one folder, in name order."""
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES)


def _box_area(person: Person) -> int:
    x1, y1, x2, y2 = person.bounding_box_xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def embed_photos(
    paths: list[Path], send_url: str, recv_url: str, config: dict, min_score: float, out_of_frame_tolerance: float
) -> dict[Path, tuple[np.ndarray | None, str | None]]:
    """Read every photo through the runner: its embedding, or None and the reason it was discarded."""
    out: dict[Path, tuple[np.ndarray | None, str | None]] = {}
    with PersonReader(send_url, recv_url, config) as reader:
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                out[path] = (None, "not an image")
                continue
            people = reader.people(image, min_score)
            if not people:
                out[path] = (None, "no person detected")
                continue
            largest = max(people, key=_box_area)
            embedding, gate = embed_person(largest, image.shape[:2], out_of_frame_tolerance)
            out[path] = (embedding, DISCARD_REASONS.get(gate))
    return out


def build_buckets(root: Path, folders: dict[str, list[Path]], cache: EnrollmentCache) -> dict[str, Bucket]:
    """One bucket per folder from the cached embeddings: usable rows, their paths and the discarded photos."""
    buckets = {}
    for name, paths in folders.items():
        usable, relative, discarded = [], [], []
        for path in paths:
            key = cache.key(root, path)
            embedding = cache.embeddings[key]
            if np.isnan(embedding).any():
                discarded.append((str(path.relative_to(root)), cache.reasons.get(key, "skeleton incomplete")))
            else:
                usable.append(embedding)
                relative.append(str(path.relative_to(root)))
        rows = np.vstack(usable).astype(np.float32) if usable else np.empty((0, EMBEDDING_SIZE), np.float32)
        buckets[name] = Bucket(name=name, embeddings=rows, photos=tuple(relative), found=len(paths), discarded=tuple(discarded))
    return buckets
