# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time

import requests

from arduino.app_internal.core import get_brick_config, get_brick_configured_model, resolve_address
from arduino.app_utils import AppError, Logger, brick

logger = Logger("LanguageTranslation")

# Keep each HTTP request below the device's 30 s socket timeout.
TRANSLATE_TIMEOUT_SECONDS = 25
MODELS_TIMEOUT_SECONDS = 10
CLOSE_TIMEOUT_SECONDS = 10

# Fallback batch size per request when the model does not advertise max_batch_size.
DEFAULT_MAX_BATCH_SIZE = 8


class TranslationError(AppError):
    """Base class for translation errors."""


class TranslationUnavailableError(TranslationError):
    """Raised when the translation service is unreachable or a request times out."""


class TranslationNotSupportedError(TranslationError):
    """Raised when the configured model does not translate the requested language pair."""


@brick
class LanguageTranslation:
    """Language translation brick for offline text-to-text translation using the local translation service."""

    _APP_SERVICE_NAME = "audio-analytics-runner"

    # The device runs a single translation engine shared by every brick in the app,
    # so requests, warmup and engine release are coordinated across instances:
    # concurrent requests would queue server-side while wasting their own socket
    # timeout, and /translations/close unloads the engine for everyone.
    _request_lock = threading.Lock()
    _shared_state_lock = threading.Lock()
    _started_instances = 0
    _resident_model: str | None = None

    def __init__(self, source: str | None = None, target: str | None = None):
        """Initialize the LanguageTranslation brick.

        The translation model comes from the brick configuration (overridable per-app
        in `app.yaml`). Fixed-pair models (e.g. OpusMT) impose the language pair, so
        `source` and `target` can be omitted entirely; they only matter for models
        supporting multiple language pairs, or as an explicit consistency check.

        Args:
            source (str, optional): Default source language code (ISO 639-1, e.g. "en").
            target (str, optional): Default target language code (ISO 639-1, e.g. "es").
        """
        self.api_host = resolve_address(self._APP_SERVICE_NAME)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = 8085
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"

        logger.debug(f"Initialized LanguageTranslation with API base URL: {self.api_base_url}")

        # Resolve the model: app.yaml override (per-brick `model:`) takes precedence over the brick default.
        brick_config = get_brick_config(self.__class__) or {}
        brick_id = brick_config.get("id")
        override = get_brick_configured_model(brick_id) if brick_id else None
        model_name = override or brick_config.get("model")
        if not model_name:
            raise RuntimeError("No translation model configured for the LanguageTranslation brick.")
        self._configured_model = model_name

        # The service compares language codes case-sensitively against lowercase model codes.
        self.source = source.lower() if source else None
        self.target = target.lower() if target else None

        self._models_lock = threading.Lock()
        self._models: list[dict] | None = None
        self._started = False

    def start(self):
        """Start the LanguageTranslation brick, preloading the translation model when the language pair is determined."""
        with LanguageTranslation._shared_state_lock:
            LanguageTranslation._started_instances += 1
            self._started = True
        self._warmup()

    def stop(self):
        """Stop the LanguageTranslation brick.

        The translation model is released on the device only when this is the last
        started LanguageTranslation brick, since the engine is shared by all of them.
        """
        with LanguageTranslation._shared_state_lock:
            if self._started:
                self._started = False
                LanguageTranslation._started_instances -= 1
            release = LanguageTranslation._started_instances == 0
            if release:
                LanguageTranslation._resident_model = None
        if release:
            self._close_remote_session()
        else:
            logger.debug("Keeping the translation engine loaded: other LanguageTranslation bricks are still started")

    def translate(self, text: str, source: str | None = None, target: str | None = None) -> str:
        """
        Translate a text with the configured model.

        Args:
            text (str): The text to translate. Empty or whitespace-only text is
                returned unchanged without contacting the service.
            source (str, optional): Source language code, overriding the constructor default.
                Ignored when the configured model translates a single language pair.
            target (str, optional): Target language code, overriding the constructor default.
                Ignored when the configured model translates a single language pair.

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the configured model supports multiple language pairs and the
                requested pair is not narrowed down to one.
            TranslationNotSupportedError: If the configured model does not translate the
                requested language pair.
            TranslationUnavailableError: If the service is unreachable or the request times out.
            TranslationError: If the model is not available on the device or the service
                fails to translate.
        """
        return self.translate_batch([text], source=source, target=target)[0]

    def translate_batch(self, texts: list[str], source: str | None = None, target: str | None = None) -> list[str]:
        """
        Translate multiple texts with the configured model.

        Texts are translated sequentially by the service, so latency grows linearly
        with the batch size. Large batches are split into multiple requests to stay
        within the service timeout.

        Args:
            texts (list[str]): The texts to translate. Empty or whitespace-only texts
                are returned unchanged without being sent to the service.
            source (str, optional): Source language code, overriding the constructor default.
                Ignored when the configured model translates a single language pair.
            target (str, optional): Target language code, overriding the constructor default.
                Ignored when the configured model translates a single language pair.

        Returns:
            list[str]: The translated texts, index-aligned with the input.

        Raises:
            ValueError: If a text is not a string, or the configured model supports multiple
                language pairs and the requested pair is not narrowed down to one.
            TranslationNotSupportedError: If the configured model does not translate the
                requested language pair.
            TranslationUnavailableError: If the service is unreachable or the request times out.
            TranslationError: If the model is not available on the device or the service
                fails to translate.
        """
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings.")

        results = list(texts)
        pending = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not pending:
            return results

        model_name, source, target, max_batch_size = self._resolve_request(source, target)

        started_at = time.perf_counter()
        for offset in range(0, len(pending), max_batch_size):
            chunk = pending[offset : offset + max_batch_size]
            translated = self._translate_request([text for _, text in chunk], model_name, source, target)
            for (index, _), translated_text in zip(chunk, translated):
                results[index] = translated_text
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"Translated {len(pending)} text(s) '{source}' -> '{target}' in {elapsed_ms:.2f} ms (model={model_name})")

        return results

    def supported_language_pairs(self) -> list[tuple[str, str]]:
        """
        List the language pairs supported by the configured model.

        Fixed-pair models return a single pair; multilingual models return one pair
        per supported direction.

        Returns:
            list[tuple[str, str]]: (source, target) language code pairs.

        Raises:
            TranslationUnavailableError: If the service is unreachable.
            TranslationError: If the model is not available on the device or the service
                fails to list the models.
        """
        return self._collect_pairs(self._model_entries())

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Catalog ids and runner-reported names differ only by separators
        (e.g. catalog `opusmt-en-zh` vs runner `opus_mt_en_zh`), so compare them
        stripped of `-` and `_`."""
        return name.replace("-", "").replace("_", "").lower()

    def _model_entries(self) -> list[dict]:
        """Runner entries for the configured model, refreshing the catalog once on miss.

        The runner reports one entry per language pair, so a fixed-pair model has a
        single entry while a multilingual model surfaces several sharing its name.
        """
        entries = self._find_entries(self._get_models())
        if not entries:
            entries = self._find_entries(self._refresh_models())
        if not entries:
            available = ", ".join(sorted({m["name"] for m in self._get_models() if m.get("name")})) or "none"
            raise TranslationError(f"Translation model '{self._configured_model}' is not available on the runner (available: {available}).")
        return entries

    def _find_entries(self, models: list[dict]) -> list[dict]:
        wanted = self._normalize_model_name(self._configured_model)
        return [entry for entry in models if entry.get("name") and self._normalize_model_name(entry["name"]) == wanted]

    def _resolve_request(self, source: str | None, target: str | None) -> tuple[str, str, str, int]:
        """Resolve the runner model name, language pair and batch size for a translate call.

        The requested pair narrows down the configured model's entries; when the model
        declares a single pair no languages are needed. Entries without language codes
        act as wildcards and require a fully specified pair.
        """
        source = (source or self.source or "").lower() or None
        target = (target or self.target or "").lower() or None

        matches = []
        for entry in self._model_entries():
            entry_source, entry_target = self._pair_codes(entry)
            if entry_source and entry_target:
                if (source is None or source == entry_source) and (target is None or target == entry_target):
                    matches.append((entry, entry_source, entry_target))
            elif source and target:
                matches.append((entry, source, target))

        if not matches:
            requested = f"'{source or '?'}' -> '{target or '?'}'"
            supported = ", ".join(f"'{s}' -> '{t}'" for s, t in self.supported_language_pairs()) or "unknown"
            raise TranslationNotSupportedError(f"Model '{self._configured_model}' does not translate {requested} (supported: {supported}).")

        # Ambiguity only exists across distinct pairs, which the caller can narrow down.
        # Duplicate entries for the same pair are equivalent: take the first one.
        distinct_pairs = {(s, t) for _, s, t in matches}
        if len(distinct_pairs) > 1:
            pairs = ", ".join(f"'{s}' -> '{t}'" for s, t in sorted(distinct_pairs))
            raise ValueError(f"Model '{self._configured_model}' supports multiple language pairs ({pairs}): specify source and target.")

        entry, source, target = matches[0]
        max_batch_size = (entry.get("parameters") or {}).get("max_batch_size") or DEFAULT_MAX_BATCH_SIZE
        return entry["name"], source, target, max_batch_size

    @staticmethod
    def _pair_codes(entry: dict) -> tuple[str | None, str | None]:
        return (
            (entry.get("source_language") or {}).get("code"),
            (entry.get("target_language") or {}).get("code"),
        )

    @staticmethod
    def _collect_pairs(models: list[dict]) -> list[tuple[str, str]]:
        pairs = []
        for entry in models:
            source, target = LanguageTranslation._pair_codes(entry)
            if source and target and (source, target) not in pairs:
                pairs.append((source, target))
        return pairs

    def _get_models(self) -> list[dict]:
        with self._models_lock:
            models = self._models
        if models is not None:
            return models
        return self._refresh_models()

    def _refresh_models(self) -> list[dict]:
        try:
            response = requests.get(f"{self.api_base_url}/translations/models", timeout=MODELS_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as e:
            raise TranslationUnavailableError(f"Translation service unreachable: {e}") from None

        if response.status_code != 200:
            raise TranslationError(self._error_detail(response, "Failed to fetch translation models."))

        models = response.json() or []
        with self._models_lock:
            self._models = models
        return models

    def _translate_request(self, texts: list[str], model_name: str, source: str, target: str) -> list[str]:
        payload = {
            "text": texts,
            "model": model_name,
            "source_language": source,
            "target_language": target,
            "keep_alive": True,  # Inert for T2T in the current service build, but forward-compatible
        }
        with self._request_lock:
            try:
                response = requests.post(
                    f"{self.api_base_url}/translations/translate",
                    json=payload,
                    timeout=TRANSLATE_TIMEOUT_SECONDS,
                )
            except requests.exceptions.RequestException as e:
                raise TranslationUnavailableError(f"Translation service unreachable: {e}") from None

        # The service reports server-side failures as 400 too, so surface the message regardless of status.
        if response.status_code != 200:
            raise TranslationError(self._error_detail(response, "Failed to translate text."))

        try:
            translations = response.json().get("translations") or []
        except ValueError:
            raise TranslationError("Malformed response from translation API.") from None

        if len(translations) != len(texts):
            raise TranslationError(f"Translation API returned {len(translations)} result(s) for {len(texts)} text(s).")

        with LanguageTranslation._shared_state_lock:
            LanguageTranslation._resident_model = self._normalize_model_name(model_name)

        return [entry.get("translated_text", "") for entry in translations]

    @staticmethod
    def _error_detail(response, fallback: str) -> str:
        """Extract the error message, handling both the API envelope and the schema-validator shape."""
        try:
            body = response.json()
        except ValueError:
            return response.text or fallback
        error = body.get("error") or {}
        return error.get("message") or body.get("message") or fallback

    def _warmup(self) -> None:
        """Best-effort warmup: translate a short text so the inference container loads
        the translation model before the first real translate().

        Skipped when another brick's model is already loaded on the device: warming up
        would evict it, and this brick's first translate() pays the load anyway.
        """
        try:
            model_name, source, target, _ = self._resolve_request(None, None)
        except ValueError:
            logger.debug("Skipping translation warmup: the language pair is not determined")
            return
        except Exception as e:
            logger.warning(f"Translation warmup failed: {e}")
            return

        with LanguageTranslation._shared_state_lock:
            resident = LanguageTranslation._resident_model
        if resident is not None and resident != self._normalize_model_name(model_name):
            logger.debug("Skipping translation warmup: another brick's model is loaded on the device")
            return

        started_at = time.perf_counter()
        try:
            self._translate_request(["ok"], model_name, source, target)
        except Exception as e:
            logger.warning(f"Translation warmup failed: {e}")
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"Translation warmup completed in {elapsed_ms:.2f} ms")

    def _close_remote_session(self) -> None:
        """Release the translation model on the device. The response body is plain text: check the status only."""
        try:
            response = requests.post(f"{self.api_base_url}/translations/close", timeout=CLOSE_TIMEOUT_SECONDS)
            if response.status_code != 200:
                logger.warning(f"Failed to close translation session: status_code={response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to close translation session: {e}")
