# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import requests

from arduino.app_bricks.language_translation import (
    LanguageTranslation,
    TranslationError,
    TranslationNotSupportedError,
    TranslationUnavailableError,
)
from arduino.app_utils import App

# Runner-reported catalog: one entry per language pair. Fixed-pair models
# (the OpusMT family) have a single entry each.
MODELS = [
    {
        "name": "opus_mt_en_es",
        "source_language": {"code": "en", "name": "English"},
        "target_language": {"code": "es", "name": "Spanish"},
        "parameters": {"max_text_length": 512, "max_batch_size": 2},
    },
    {
        "name": "opus_mt_es_en",
        "source_language": {"code": "es", "name": "Spanish"},
        "target_language": {"code": "en", "name": "English"},
    },
]

# A hypothetical multilingual model surfaces one entry per pair, sharing its name.
MULTILINGUAL_MODELS = [
    {
        "name": "nllb_200",
        "source_language": {"code": "en", "name": "English"},
        "target_language": {"code": "es", "name": "Spanish"},
    },
    {
        "name": "nllb_200",
        "source_language": {"code": "es", "name": "Spanish"},
        "target_language": {"code": "en", "name": "English"},
    },
]


@pytest.fixture(autouse=True)
def reset_shared_engine_state(monkeypatch):
    """Bricks coordinate on class-level state; isolate it per test."""
    monkeypatch.setattr(LanguageTranslation, "_started_instances", 0)
    monkeypatch.setattr(LanguageTranslation, "_resident_model", None)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON body")
        return self._json_data


def echo_translation(url, json, **kwargs):
    """Default translate handler: prefixes each text with the target language."""
    translations = [
        {
            "translated_text": f"{json['target_language']}:{text}",
            "source_language": json["source_language"],
            "target_language": json["target_language"],
        }
        for text in json["text"]
    ]
    return FakeResponse(json_data={"translations": translations})


def make_translator(monkeypatch, model="opusmt-en-es", models=MODELS, post_response=echo_translation, **kwargs):
    get_calls = []
    post_calls = []

    def get(url, **request_kwargs):
        get_calls.append(url)
        return FakeResponse(json_data=models)

    def post(url, json=None, **request_kwargs):
        post_calls.append({"url": url, "json": json})
        if url.endswith("/translations/close"):
            return FakeResponse(text="Translation session successfully closed.")
        return post_response(url, json, **request_kwargs)

    module = "arduino.app_bricks.language_translation.local_translation"
    monkeypatch.setattr(f"{module}.requests.get", get)
    monkeypatch.setattr(f"{module}.requests.post", post)
    monkeypatch.setattr(f"{module}.get_brick_config", lambda cls: {"id": "arduino:language_translation", "model": model})
    monkeypatch.setattr(f"{module}.get_brick_configured_model", lambda brick_id: None)

    translator = LanguageTranslation(**kwargs)
    App.unregister(translator)
    return translator, get_calls, post_calls


def test_translate_uses_pair_imposed_by_fixed_pair_model(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch)

    # No languages configured anywhere: the configured model determines the pair,
    # and the request carries the runner-reported name, not the catalog id.
    result = translator.translate("Hello world")

    assert result == "es:Hello world"
    payload = post_calls[0]["json"]
    assert payload["model"] == "opus_mt_en_es"
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "es"
    assert payload["keep_alive"] is True


def test_translate_accepts_matching_languages_and_lowercases_them(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch, source="EN", target="ES")

    translator.translate("Hello")

    payload = post_calls[0]["json"]
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "es"


def test_translate_rejects_pair_not_served_by_configured_model(monkeypatch):
    translator, _, _ = make_translator(monkeypatch)

    with pytest.raises(TranslationNotSupportedError):
        translator.translate("Adiós", source="es", target="en")  # Served by another model, not the configured one


def test_translate_with_unknown_configured_model_refreshes_once_then_raises(monkeypatch):
    translator, get_calls, _ = make_translator(monkeypatch, model="opusmt-en-fr")

    with pytest.raises(TranslationError, match="not available"):
        translator.translate("Hello")

    assert len(get_calls) == 2  # Initial fetch plus one refresh on miss


def test_multilingual_model_requires_language_pair(monkeypatch):
    translator, _, _ = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS)

    with pytest.raises(ValueError, match="multiple language pairs"):
        translator.translate("Hello")


def test_multilingual_model_selects_entry_from_language_pair(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS)

    result = translator.translate("Adiós", source="es", target="en")

    assert result == "en:Adiós"
    payload = post_calls[0]["json"]
    assert payload["model"] == "nllb_200"
    assert payload["source_language"] == "es"
    assert payload["target_language"] == "en"


def test_multilingual_model_accepts_partial_pair_when_unambiguous(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS)

    translator.translate("Hello", target="es")

    payload = post_calls[0]["json"]
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "es"


def test_duplicate_entries_for_same_pair_resolve_to_first_without_error(monkeypatch):
    duplicated = [
        {
            "name": "opus_mt_en_es",
            "source_language": {"code": "en", "name": "English"},
            "target_language": {"code": "es", "name": "Spanish"},
            "parameters": {"max_batch_size": 2},
        },
        {
            "name": "opus_mt_en_es",
            "source_language": {"code": "en", "name": "English"},
            "target_language": {"code": "es", "name": "Spanish"},
            "parameters": {"max_batch_size": 5},
        },
    ]
    translator, _, post_calls = make_translator(monkeypatch, models=duplicated)

    results = translator.translate_batch(["a", "b", "c"])

    assert results == ["es:a", "es:b", "es:c"]
    # First entry wins: its max_batch_size of 2 drives the request splitting.
    assert [call["json"]["text"] for call in post_calls] == [["a", "b"], ["c"]]


def test_wildcard_entry_without_codes_requires_explicit_pair(monkeypatch):
    codeless = [{"name": "open_translator"}]
    translator, _, post_calls = make_translator(monkeypatch, model="open-translator", models=codeless)

    with pytest.raises(TranslationNotSupportedError):
        translator.translate("Hello")

    translator.translate("Hello", source="en", target="es")

    payload = post_calls[0]["json"]
    assert payload["model"] == "open_translator"
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "es"


def test_translate_empty_text_skips_service(monkeypatch):
    translator, get_calls, post_calls = make_translator(monkeypatch)

    assert translator.translate("") == ""
    assert translator.translate("   ") == "   "
    assert get_calls == []
    assert post_calls == []


def test_translate_batch_is_index_aligned_and_preserves_empty_texts(monkeypatch):
    translator, _, _ = make_translator(monkeypatch)

    results = translator.translate_batch(["Hello", "", "World"])

    assert results == ["es:Hello", "", "es:World"]


def test_translate_batch_splits_requests_by_model_max_batch_size(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch)

    results = translator.translate_batch(["a", "b", "c"])

    assert results == ["es:a", "es:b", "es:c"]
    assert [call["json"]["text"] for call in post_calls] == [["a", "b"], ["c"]]


def test_translate_batch_rejects_non_string_texts(monkeypatch):
    translator, _, _ = make_translator(monkeypatch)

    with pytest.raises(ValueError):
        translator.translate_batch(["Hello", 42])


def test_models_are_cached_across_calls(monkeypatch):
    translator, get_calls, _ = make_translator(monkeypatch)

    translator.translate("Hello")
    translator.translate("World")

    assert len(get_calls) == 1


def test_translate_surfaces_api_error_envelope(monkeypatch):
    error = FakeResponse(status_code=400, json_data={"error": {"message": "engine exploded", "type": "server_error"}})
    translator, _, _ = make_translator(monkeypatch, post_response=lambda url, json, **kwargs: error)

    with pytest.raises(TranslationError, match="engine exploded"):
        translator.translate("Hello")


def test_translate_surfaces_schema_validator_error_shape(monkeypatch):
    error = FakeResponse(status_code=400, json_data={"message": "request.body should have required property"})
    translator, _, _ = make_translator(monkeypatch, post_response=lambda url, json, **kwargs: error)

    with pytest.raises(TranslationError, match="required property"):
        translator.translate("Hello")


def test_translate_raises_unavailable_on_connection_error(monkeypatch):
    def post_response(url, json, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    translator, _, _ = make_translator(monkeypatch, post_response=post_response)

    with pytest.raises(TranslationUnavailableError):
        translator.translate("Hello")


def test_translate_raises_on_mismatched_result_count(monkeypatch):
    short = FakeResponse(json_data={"translations": [{"translated_text": "Hola"}]})
    translator, _, _ = make_translator(monkeypatch, post_response=lambda url, json, **kwargs: short)

    with pytest.raises(TranslationError):
        translator.translate_batch(["Hello", "World"])


def test_supported_language_pairs_lists_configured_model_pairs_only(monkeypatch):
    fixed, _, _ = make_translator(monkeypatch)
    assert fixed.supported_language_pairs() == [("en", "es")]

    multilingual, _, _ = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS)
    assert multilingual.supported_language_pairs() == [("en", "es"), ("es", "en")]


def test_start_warms_up_fixed_pair_model_without_configured_languages(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch)

    translator.start()

    assert len(post_calls) == 1
    assert post_calls[0]["json"]["model"] == "opus_mt_en_es"


def test_start_skips_warmup_when_pair_is_undetermined(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS)

    translator.start()

    assert post_calls == []


def test_start_warms_up_multilingual_model_with_configured_languages(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch, model="nllb-200", models=MULTILINGUAL_MODELS, source="es", target="en")

    translator.start()

    assert len(post_calls) == 1
    assert post_calls[0]["json"]["source_language"] == "es"


def test_start_warmup_failure_is_logged_not_raised(monkeypatch):
    def post_response(url, json, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    translator, _, _ = make_translator(monkeypatch, post_response=post_response)

    logged_warnings = []
    monkeypatch.setattr(
        "arduino.app_bricks.language_translation.local_translation.logger.warning",
        lambda msg, *args, **kwargs: logged_warnings.append(msg),
    )

    translator.start()

    assert any("warmup failed" in msg for msg in logged_warnings)


def test_stop_closes_remote_session_and_tolerates_plain_text_response(monkeypatch):
    translator, _, post_calls = make_translator(monkeypatch)

    translator.stop()

    assert [call["url"] for call in post_calls] == [translator.api_base_url + "/translations/close"]


def test_second_brick_skips_warmup_when_another_model_is_resident(monkeypatch):
    first, _, first_posts = make_translator(monkeypatch)
    first.start()
    assert [call["json"]["model"] for call in first_posts] == ["opus_mt_en_es"]

    second, _, second_posts = make_translator(monkeypatch, model="opusmt-es-en")
    second.start()

    assert second_posts == []  # Warming up would evict the first brick's model


def test_second_brick_with_same_model_still_warms_up(monkeypatch):
    first, _, _ = make_translator(monkeypatch)
    first.start()

    second, _, second_posts = make_translator(monkeypatch)
    second.start()

    assert [call["json"]["model"] for call in second_posts] == ["opus_mt_en_es"]


def test_engine_is_released_only_when_last_brick_stops(monkeypatch):
    first, _, first_posts = make_translator(monkeypatch)
    second, _, second_posts = make_translator(monkeypatch, model="opusmt-es-en")
    first.start()
    second.start()

    first.stop()
    assert not any(call["url"].endswith("/translations/close") for call in first_posts)

    second.stop()
    assert [call["url"] for call in second_posts if call["url"].endswith("/translations/close")] == [second.api_base_url + "/translations/close"]


def test_stop_is_idempotent_and_does_not_release_for_other_started_bricks(monkeypatch):
    first, _, first_posts = make_translator(monkeypatch)
    second, _, _ = make_translator(monkeypatch, model="opusmt-es-en")
    first.start()
    second.start()

    first.stop()
    first.stop()  # Repeated stop must not decrement the count below the started bricks

    assert not any(call["url"].endswith("/translations/close") for call in first_posts)
