# Language Translation (T2T) Brick

The `LanguageTranslation` brick provides a completely offline text-to-text translation solution for Arduino Apps. It translates text between languages using translation models running locally on the device, ensuring privacy and eliminating network dependencies.

## Features

- **Offline Operation:** All translations are performed locally by the device's translation service.
- **Simple API:** With a fixed-pair model (e.g. OpusMT, the default) the language pair is imposed by the model, so `translate()` needs only the text. Models supporting multiple language pairs take the pair from the constructor or per call.
- **Batch Translation:** `translate_batch()` translates a list of texts and returns the results index-aligned with the input. Large batches are automatically split into multiple requests to stay within the service timeout.
- **Model Selection:** The translation model is configured in `brick_config.yaml` and can be overridden per-app in `app.yaml` (e.g. `opusmt-en-zh` for English to Chinese). Use `supported_language_pairs()` to see what the configured model offers.
- **Model Preloading:** The translation model is loaded at brick start so the first `translate()` call is fast.

Fixed-pair models like the OpusMT family translate a single language pair in one direction (e.g. English to Chinese).

## Code Example and Usage

This example translates text from English to Chinese using the default `opusmt-en-zh` model.

```python
from arduino.app_bricks.language_translation import LanguageTranslation
from arduino.app_utils import App


translator = LanguageTranslation()


def runner():
    print(translator.translate("Hello world"))  # "你好世界"


App.run(user_loop=runner)
```

### Batch translation

```python
from arduino.app_bricks.language_translation import LanguageTranslation

translator = LanguageTranslation()
translated = translator.translate_batch(["Hello world", "How are you?"])
# ["你好世界", "你好吗？"]
```

### Multilingual models

Models supporting multiple language pairs need the pair to be specified, either once at construction or per call. Fixed-pair models ignore the distinction: passing their own pair is accepted, any other pair raises `TranslationNotSupportedError`.

```python
from arduino.app_bricks.language_translation import LanguageTranslation

translator = LanguageTranslation(source="en", target="es")
translator.translate("Hello")                              # Default pair: "Hola"
translator.translate("Adiós", source="es", target="en")    # Per-call pair: "Goodbye"
```

Note: switching the language pair between calls makes the device unload and reload the translation engine, adding around a second of latency. Group calls by language pair when possible.

### Multiple translation bricks

Several `LanguageTranslation` bricks in the same app coordinate on the device's single translation engine: requests are serialized across bricks, the engine is released only when the last brick stops, and only one model is loaded at a time — each model or pair switch reloads the engine, adding around a second of latency.

### Discover the supported language pairs

```python
from arduino.app_bricks.language_translation import LanguageTranslation

translator = LanguageTranslation()
print(translator.supported_language_pairs())  # [("en", "zh")]
```

## Configuration

`LanguageTranslation(source=None, target=None)`: `source` and `target` are ISO 639-1 language codes (e.g. `"en"`, `"es"`) setting the default translation pair. Both are unnecessary with fixed-pair models, whose pair is imposed by the model itself; for multilingual models they can also be omitted here and passed per call to `translate()` / `translate_batch()` instead.

The model is set by the `model` field in `brick_config.yaml` and can be overridden per-app in `app.yaml`.

## Interaction with ASR and TTS

The translation service shares the device's audio pipeline with the ASR and TTS bricks. A translation request takes over the pipeline, so avoid translating while a live transcription or speech session must stay active. The translation model stays resident in device memory once loaded; it is released when the last `LanguageTranslation` brick is stopped.

## Errors

- `TranslationNotSupportedError`: the configured model does not translate the requested language pair. Check `supported_language_pairs()`.
- `TranslationUnavailableError`: the translation service is unreachable or the request timed out.
- `TranslationError`: base class for all of the above, also raised when the configured model is not installed on the device or the service fails to translate.
- `ValueError`: the configured model supports multiple language pairs and the requested pair was not narrowed down to one.
