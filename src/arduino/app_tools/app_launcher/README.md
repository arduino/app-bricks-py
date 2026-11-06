# arduino-app-launcher

Runs inside the container of one app, started by `/run.sh serve` of `python-apps-base` as `python -m arduino.app_tools.app_launcher` on the interpreter of the app. It imports the libraries once and forks the app on each start request, so a start after an edit skips the interpreter startup and the imports. The container stays up when the app stops. The daemon is its client; the protocol is in `protocol.py`.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, one JSON object per line, one request and one response per connection.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"/app","env":{"K":"V"}}` | `{"ok":true,"pid":N,"started_at":"<iso>","runner":"python","timings":{"stop_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"child":{..} or null,"last":{..} or null,"venv":..,"preload":{"modules","loaded","failed"},"version":..,"uptime_s":..}` |

A start while an app runs stops it first. `env` is laid over the container environment. Errors are `{"ok":false,"error":{"code":..,"message":..}}`, with `code` one of `bad_request`, `not_found`, `stale` (the dependencies changed since run.sh installed them: the daemon restarts the container), `busy` (shutting down) and `internal`.

## Behaviour

- The child runs `python/main.py` as `__main__` from the app folder, with the container environment, the request `env`, `PYTHONUNBUFFERED=1`, `<app>/bricks` on `PYTHONPATH` and the app venv active when the launcher runs in it. A Streamlit app, `arduino:streamlit_ui` in its app.yaml, runs through `streamlit.web.bootstrap` on port 7000.
- A stop sends SIGTERM to the app process group, SIGKILL after `APP_LAUNCHER_STOP_TIMEOUT_S`, then kills what the app left behind when the launcher tops a pid namespace of its own, as in its container.
- The launcher is single-threaded and forks between two turns of its loop, never inside a callback. The preload runs when idle, one module at a time: a start never waits for it. On SIGTERM it stops the app and exits with 128 plus the signal, as a stopped app does.
- The app writes to the container stdout and stderr; the launcher logs to stderr with the `app-launcher:` prefix.

## Environment

| Variable | Default | |
|---|---|---|
| `APP_DIR` | unset | The app folder run.sh prepared, the only one served; unset, any folder is served |
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils`, the camera, microphone and speaker backends, `arduino.app_bricks.web_ui` | Modules imported when idle; in an app venv `streamlit.web.bootstrap` is added when the venv has it |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL |
| `APP_LAUNCHER_LOG_LEVEL` | `WARNING` | Log level of the launcher's own lines |
