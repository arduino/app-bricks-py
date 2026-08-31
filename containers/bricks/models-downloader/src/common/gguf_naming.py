# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Names for the GGUF models of the llamacpp tree, decided by the curated catalog.

Different Hugging Face repositories can publish identically named GGUF files, so a
model cannot be named after its file alone: two separate downloads would share one
name — one listing id, one models.ini section — and delete, status and size would all
act on whichever file happened to win.

``gguf_model_name`` names a file by whether models-list.yaml declares it:

- A **curated** file (its location matches a catalog declaration) keeps the file stem.
  Curated ids are fixed ``llamacpp:<stem>`` keys in models-list.yaml, referenced from
  installed apps, so their names are backward compatible forever.
- An **ad-hoc** file is named by its tree-relative path without the extension, which
  carries the repository (e.g. "unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M"). The name is
  unique by construction and stable for the whole life of the install — it never
  changes when a same-named file is downloaded from another repository, and an
  impostor named like a curated model can never answer to the curated name.

The same rule must name a model everywhere it appears: the listing id
(``llamacpp:<name>``), the models.ini section (``<name>``) and the fallback metadata
id. The LLM brick resolves ``llamacpp:<name>`` to the model llama-server serves under
the section ``<name>``, so these may never drift apart. The runners' standalone
configure-llamacpp.py scripts replicate this naming and must be kept in sync.

The catalog is baked into every image that names models (the models-downloader and
both llamacpp runners), so a missing or unreadable copy is a build defect: it degrades
to "no declarations", which names every file by its path and makes curated models fail
loudly instead of serving under an ambiguous stem.
"""

from common.models_list import MODELS_LIST_PATH, _iter_platform_variables, get_model_subdir, load_models_list

GGUF_SUFFIX = ".gguf"

# The subdirectory of the models mount that holds GGUF models, and the id namespace
# derived from it. The declarations below are relative to this directory.
LLAMACPP_SUBDIR = "llamacpp"


def gguf_basename(model_url):
    """The GGUF file name *model_url* pins, or None when it does not pin one.

    Works on both syntaxes hf_downloader accepts: the basename of a file URL, or a
    compact key whose quantization field is a full file name
    ("org/repo:model-Q4_0.gguf"). A key naming only a quantization pins no file.
    """
    base = (model_url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return base if base.endswith(GGUF_SUFFIX) else None


def declared_gguf_files(models):
    """The file location every llamacpp-backed models-list.yaml entry declares.

    Returns:
        ``(model_directory, filename, model_id)`` tuples, deduplicated across the
        per-board platform repeats. ``filename`` is None when the entry's model_url
        pins no file (a compact key naming a quantization); any GGUF inside the
        directory then counts as declared.
    """
    declarations = []
    for model_id, _model_data, _platform, variables in _iter_platform_variables(models):
        if get_model_subdir(variables.get("models_repository", "")) != LLAMACPP_SUBDIR:
            continue
        directory = (variables.get("model_directory") or "").strip("/")
        if not directory:
            continue
        declaration = (directory, gguf_basename(variables.get("model_url", "")), model_id)
        if declaration not in declarations:
            declarations.append(declaration)
    return declarations


def catalog_gguf_declarations(models_list_path=MODELS_LIST_PATH):
    """``declared_gguf_files`` of the catalog baked into the image; () when unusable."""
    try:
        return declared_gguf_files(load_models_list(models_list_path))
    except Exception:  # noqa: BLE001 - see the module docstring: degrade, never crash
        return ()


def declaration_covers(directory, declared_name, rel_dir, filename):
    """Whether a declaration covers the GGUF at *rel_dir*/*filename*.

    The file may sit below the declared directory (some repositories nest their
    files in per-quantization folders). A declaration that pins no file name covers
    any GGUF in its directory; an unknown *filename* (a pending download whose
    marker names no file) is only covered by those.
    """
    if rel_dir != directory and not rel_dir.startswith(directory + "/"):
        return False
    return declared_name is None or declared_name == filename


def gguf_model_name(rel_path, declarations):
    """The name identifying the GGUF at *rel_path* (tree-relative, posix separators).

    The file stem when *declarations* covers the file — see the module docstring —
    the relative path without its extension otherwise. Callers pass only main model
    files: mmproj companions belong to the model in the same directory and never
    name one.
    """
    rel_dir, _, filename = rel_path.rpartition("/")
    if any(declaration_covers(directory, declared_name, rel_dir, filename) for directory, declared_name, _model_id in declarations):
        return filename[: -len(GGUF_SUFFIX)]
    return rel_path[: -len(GGUF_SUFFIX)]
