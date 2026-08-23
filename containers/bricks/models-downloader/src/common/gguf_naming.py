# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unique names for the GGUF models sharing one directory tree.

Different Hugging Face repositories can publish identically named GGUF files, so a
model cannot be named after its file alone: two separate downloads would share one
name — one listing id, one models.ini section — and delete, status and size would all
act on whichever file happened to win.

``gguf_model_names`` gives every file in the tree a distinct name: the file stem when
no other file shares it (the historical name, so existing references keep resolving),
the tree-relative path without its extension otherwise (which carries the repository,
e.g. "unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M").

The same map must name a model everywhere it appears: the listing id
(``llamacpp:<name>``), the models.ini section (``<name>``) and the fallback metadata
id. The LLM brick resolves ``llamacpp:<name>`` to the model llama-server serves under
the section ``<name>``, so these may never drift apart. The runners' standalone
configure-llamacpp.py scripts replicate this naming and must be kept in sync.
"""

from collections import Counter

GGUF_SUFFIX = ".gguf"


def gguf_model_names(rel_paths):
    """Map each tree-relative GGUF path (posix separators) to its model name.

    Callers pass only main model files — mmproj companions belong to the model in the
    same directory and never name one.
    """
    stems = Counter(_stem(path) for path in rel_paths)
    return {path: _stem(path) if stems[_stem(path)] == 1 else path[: -len(GGUF_SUFFIX)] for path in rel_paths}


def _stem(rel_path):
    name = rel_path.rsplit("/", 1)[-1]
    return name[: -len(GGUF_SUFFIX)] if name.endswith(GGUF_SUFFIX) else name
