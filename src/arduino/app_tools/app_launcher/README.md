# arduino-app-launcher

Runs inside the container of one app, started by `/run.sh launch` of `python-apps-base` as `python -m arduino.app_tools.app_launcher` on the interpreter of the app. It starts the app before it loads itself, so the app never waits for it, then imports the libraries once and ends when the app ends on its own, as the app did: the container lives as long as its app. A start request replaces the running app, so a start after an edit skips the interpreter startup and the imports. The daemon is its client; the protocol is in `protocol.py`.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, one JSON object per line, one request and one response per connection.

| Request | Response |
|---|---|
| `{"cmd":"start","env":{"K":"V"}}` | `{"ok":true,"pid":N,"started_at":"<iso>","runner":"python","timings":{"stop_ms":..,"fork_ms":..}}` |
| `{"cmd":"status"}` | `{"ok":true,"child":{..} or null,"last":{..} or null,"venv":..,"preload":{"modules","loaded","failed"},"version":..,"uptime_s":..}` |

A start runs the app again: the running one is stopped first. A start that leaves no app running ends the launcher: it never stays up without its app. `env` is laid over the container environment. Errors are `{"ok":false,"error":{"code":..,"message":..}}`, with `code` one of `bad_request`, `not_found`, `stale` (the dependencies changed since run.sh installed them: the app needs a fresh container), `busy` (shutting down) and `internal`.

## Behaviour

- The child runs `python/main.py` as `__main__` from the app folder, with the container environment, the request `env`, `PYTHONUNBUFFERED=1`, `<app>/bricks` on `PYTHONPATH` and the app venv active when the launcher runs in it. A Streamlit app, `arduino:streamlit_ui` in its app.yaml, runs through `streamlit.web.bootstrap` on port 7000.
- The launcher stops the app, to replace it or on its own way out, with SIGTERM to the app process group and SIGKILL after `APP_LAUNCHER_STOP_TIMEOUT_S`. It then kills what the app left behind when it tops a pid namespace of its own, as in its container.
- The launcher is single-threaded and forks between two turns of its loop, never inside a callback. The preload runs when idle, one module at a time: a start never waits for it.
- On SIGTERM the launcher stops the app and exits with 128 plus the signal, as a stopped app does. When the app ends on its own it exits as the app did: with the exit code of the app, or 128 plus the signal that killed it. It exits with 1 when it cannot start the app.
- The app writes to the container stdout and stderr; the launcher logs to stderr with the `app-launcher:` prefix.

## Environment

| Variable | Default | |
|---|---|---|
| `APP_DIR` | required | The app folder run.sh prepared, the one served: `/app` in the container |
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils`, the camera, microphone and speaker backends, `arduino.app_bricks.web_ui` | Modules imported when idle; in an app venv `streamlit.web.bootstrap` is added when the venv has it |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL |
| `APP_LAUNCHER_LOG_LEVEL` | `WARNING` | Log level of the launcher's own lines |
