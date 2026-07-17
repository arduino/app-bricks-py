# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""API/signature stability tests for the VLM inheritance chain.

These tests guard the public contract of ``VisionLanguageModel`` and the
constructor "hand-off" between it and its parent classes
(``LargeLanguageModel`` -> ``CloudLLM``).

Background: a previous regression was caused by an ``api_key`` signature
change in a parent class. ``LargeLanguageModel.__init__`` forwards
``api_key=...`` to ``CloudLLM.__init__`` by keyword, so removing/renaming
that parameter (or making it positional-only) silently broke construction.
The tests below fail loudly if such a change happens again.
"""

import inspect

import pytest

from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.vlm import VisionLanguageModel

# Parameter kinds that a caller can supply by keyword argument.
_KEYWORD_COMPATIBLE_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def _param(cls, method_name, param_name):
    """Return the ``inspect.Parameter`` for ``cls.method_name``'s ``param_name``."""
    sig = inspect.signature(getattr(cls, method_name))
    assert param_name in sig.parameters, f"{cls.__name__}.{method_name} is missing parameter '{param_name}'"
    return sig.parameters[param_name]


def _accepts_keyword(cls, method_name, param_name):
    """Whether ``param_name`` can be passed by keyword to ``cls.method_name``.

    Either the parameter is explicitly declared as keyword-compatible, or the
    method accepts ``**kwargs`` and would absorb it.
    """
    sig = inspect.signature(getattr(cls, method_name))
    if param_name in sig.parameters:
        return sig.parameters[param_name].kind in _KEYWORD_COMPATIBLE_KINDS
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _signature_fingerprint(cls, method_name):
    """A stable, comparable fingerprint of a method signature.

    Captures each parameter's name, kind and whether it has a default. Defaults
    values themselves are intentionally excluded so cosmetic default tweaks do
    not break the test, while structural changes (renames, reordering,
    additions/removals, kind changes) are caught.
    """
    sig = inspect.signature(getattr(cls, method_name))
    return [(name, p.kind, p.default is not inspect.Parameter.empty) for name, p in sig.parameters.items()]


# ---------------------------------------------------------------------------
# Constructor hand-off contract (the actual regression that occurred).
# ---------------------------------------------------------------------------


def test_cloud_llm_init_accepts_api_key_by_keyword():
    """CloudLLM must accept ``api_key`` as an explicitly named keyword parameter.

    ``LargeLanguageModel.__init__`` forwards ``api_key="api_key"`` by keyword;
    if the parent drops/renames the parameter or makes it positional-only this
    assertion fails before the runtime break reaches users.
    """
    param = _param(CloudLLM, "__init__", "api_key")
    assert param.kind in _KEYWORD_COMPATIBLE_KINDS
    # The parent historically defaults api_key from the environment (a str).
    assert param.default is not inspect.Parameter.empty
    assert isinstance(param.default, str)


def test_large_language_model_forwards_api_key_to_cloud_llm():
    """The keyword names LargeLanguageModel forwards must exist on CloudLLM."""
    # These are the keyword arguments LargeLanguageModel.__init__ passes up.
    forwarded = ["api_key", "model", "system_prompt", "temperature", "timeout", "tools", "base_url", "max_tokens"]
    for name in forwarded:
        assert _accepts_keyword(CloudLLM, "__init__", name), f"CloudLLM.__init__ cannot accept forwarded keyword '{name}'"


def test_vlm_forwards_constructor_kwargs_to_large_language_model():
    """The keyword names VisionLanguageModel forwards must exist on the parent."""
    forwarded = ["model", "system_prompt", "temperature", "max_tokens", "timeout", "tools"]
    for name in forwarded:
        assert _accepts_keyword(LargeLanguageModel, "__init__", name), f"LargeLanguageModel.__init__ cannot accept forwarded keyword '{name}'"


def test_parent_inits_accept_var_keyword_passthrough():
    """Each class forwards ``**kwargs`` upward, so parents must accept ``**kwargs``."""
    for cls in (LargeLanguageModel, CloudLLM):
        sig = inspect.signature(cls.__init__)
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), f"{cls.__name__}.__init__ must accept **kwargs"


# ---------------------------------------------------------------------------
# Public API surface of VisionLanguageModel.
# ---------------------------------------------------------------------------


EXPECTED_VLM_PUBLIC_METHODS = {
    "__init__",
    "get_client",
    "chat",
    "chat_stream",
    "stop_stream",
    "clear_memory",
    "with_memory",
}


def test_vlm_public_methods_are_present():
    for name in EXPECTED_VLM_PUBLIC_METHODS:
        assert callable(getattr(VisionLanguageModel, name, None)), f"VisionLanguageModel.{name} is missing"


# Expected signature fingerprints for the VLM public API. Update deliberately
# (with a matching changelog entry) whenever the public contract changes.
_P = inspect.Parameter
EXPECTED_VLM_SIGNATURES = {
    "__init__": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("system_prompt", _P.POSITIONAL_OR_KEYWORD, True),
        ("temperature", _P.POSITIONAL_OR_KEYWORD, True),
        ("max_tokens", _P.POSITIONAL_OR_KEYWORD, True),
        ("timeout", _P.POSITIONAL_OR_KEYWORD, True),
        ("tools", _P.POSITIONAL_OR_KEYWORD, True),
        ("model", _P.POSITIONAL_OR_KEYWORD, True),
        ("kwargs", _P.VAR_KEYWORD, False),
    ],
    "get_client": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "chat": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("message", _P.POSITIONAL_OR_KEYWORD, False),
        ("images", _P.POSITIONAL_OR_KEYWORD, True),
    ],
    "chat_stream": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("message", _P.POSITIONAL_OR_KEYWORD, False),
        ("images", _P.POSITIONAL_OR_KEYWORD, True),
    ],
    "stop_stream": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "clear_memory": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "with_memory": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("max_messages", _P.POSITIONAL_OR_KEYWORD, True),
        ("persistence", _P.POSITIONAL_OR_KEYWORD, True),
    ],
}


@pytest.mark.parametrize("method_name", sorted(EXPECTED_VLM_SIGNATURES))
def test_vlm_public_signatures_are_stable(method_name):
    assert _signature_fingerprint(VisionLanguageModel, method_name) == EXPECTED_VLM_SIGNATURES[method_name]


def test_vlm_inheritance_chain_is_intact():
    """VLM must remain a LargeLanguageModel/CloudLLM subclass for the tests above to be meaningful."""
    assert issubclass(VisionLanguageModel, LargeLanguageModel)
    assert issubclass(LargeLanguageModel, CloudLLM)
