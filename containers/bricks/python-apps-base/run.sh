#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

OPENCV_DEBUG=0

# Disable core dumps: inherited by python and any native library it loads.
# Set ENABLE_CORE_DUMPS=1 to keep them (e.g. to debug a native crash).
if [ "${ENABLE_CORE_DUMPS:-0}" != "1" ]; then
  ulimit -c 0 2>/dev/null || true
fi

if [ -z "$PYTHONUNBUFFERED" ]; then
  export PYTHONUNBUFFERED=1
fi

BASE_DIR="/app"
CACHE_DIR="${APP_CACHE_DIR:-$BASE_DIR/.cache}"
APP_YAML="$BASE_DIR/app.yaml"
PYTHON_SCRIPT="$BASE_DIR/python/main.py"
REQUIREMENTS_FILE="$BASE_DIR/python/requirements.txt"
PYTHON_LIBS_DIR="$BASE_DIR/python-libraries"
INSTALLED_REQUIREMENTS_FILE="$CACHE_DIR/installed_requirements.txt"

# Set when a dependency install fails: only `prepare` acts on it, starting the
# app stays best-effort as before.
DEPS_FAILED=0

export UV_CACHE_DIR="$CACHE_DIR/uv"

# Remove core dumps left in the app directory by a previous run, unless they
# are explicitly wanted
if [ "${ENABLE_CORE_DUMPS:-0}" != "1" ]; then
  find "$BASE_DIR" -maxdepth 1 -type f \
    \( -name 'core' -o -name 'core.[0-9]*' -o -name '*.core' \) \
    -print -delete 2>/dev/null || true
fi

if [ "${OPENCV_DEBUG:-0}" = "1" ]; then
  export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-DEBUG}"
  export OPENCV_VIDEOIO_DEBUG="${OPENCV_VIDEOIO_DEBUG:-1}"
  export GST_DEBUG="${GST_DEBUG:-3}"
  export GST_DEBUG_NO_COLOR="${GST_DEBUG_NO_COLOR:-1}"
fi

mkdir -p "$CACHE_DIR"
if [ ! -d "$CACHE_DIR/.venv" ]; then
  uv venv "$CACHE_DIR/.venv" --system-site-packages

  if [ -d "$PYTHON_LIBS_DIR" ]; then
    echo "Installing Python libraries from $PYTHON_LIBS_DIR"
    # Iterate over each .whl.installed file in the directory and revert them to .whl
    for installed_file in "$PYTHON_LIBS_DIR"/*.whl.installed; do
      if [ -f "$installed_file" ]; then
        original_file="${installed_file%.installed}"
        echo "  Restoring: $installed_file -> $original_file"
        mv "$installed_file" "$original_file"
      fi
    done
  fi
fi

. "$CACHE_DIR/.venv/bin/activate"

if [ -d "$PYTHON_LIBS_DIR" ]; then
  echo "Installing Python libraries from $PYTHON_LIBS_DIR"
  # Iterate over each .whl file in the directory
  for whl_file in "$PYTHON_LIBS_DIR"/*.whl; do
    if [ -f "$whl_file" ]; then
      uv pip install --compile-bytecode "$whl_file"
      mv "$whl_file" "$whl_file".installed
    fi
  done
fi

if [ -f "$REQUIREMENTS_FILE" ]; then
  INSTALL_DEPS=1
  REQUIREMENTS_LINES="$(grep -c '[^[:space:]]' "$REQUIREMENTS_FILE")"
  if [ -f "$INSTALLED_REQUIREMENTS_FILE" ]; then
    if cmp -s "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"; then
      echo "Requirements already installed."
      INSTALL_DEPS=0
    fi
  fi
  if [ "$INSTALL_DEPS" -gt 0 ]; then
    if [ "$REQUIREMENTS_LINES" -ne 0 ]; then
      if uv pip install --compile-bytecode -r "$REQUIREMENTS_FILE"; then
        cp "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"
      else
        DEPS_FAILED=1
      fi
    else
      cp "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"
    fi
  fi
  # clean up cache
  uv cache clean
fi

# Install custom brick requirements with caching
if [ -d "$BASE_DIR/bricks" ]; then
  for brick_dir in $BASE_DIR/bricks/*; do
    if [ -d "$brick_dir" ]; then
      brick_name=$(basename "$brick_dir")
      brick_requirements="${brick_dir}/requirements.txt"
      brick_cache_dir="$CACHE_DIR/$brick_name"
      brick_installed_requirements="$brick_cache_dir/installed_requirements.txt"
      
      if [ -f "$brick_requirements" ]; then
        mkdir -p "$brick_cache_dir"
        
        INSTALL_BRICK_DEPS=1
        BRICK_REQUIREMENTS_LINES="$(grep -c '[^[:space:]]' "$brick_requirements")"
        
        if [ -f "$brick_installed_requirements" ]; then
          if cmp -s "$brick_requirements" "$brick_installed_requirements"; then
            echo "Brick requirements for '$brick_name' already installed."
            INSTALL_BRICK_DEPS=0
          fi
        fi
        
        if [ "$INSTALL_BRICK_DEPS" -gt 0 ]; then
          if [ "$BRICK_REQUIREMENTS_LINES" -ne 0 ]; then
            echo "Installing requirements for brick: $brick_name"
            if uv pip install --compile-bytecode -r "$brick_requirements"; then
              cp "$brick_requirements" "$brick_installed_requirements"
            else
              DEPS_FAILED=1
            fi
          else
            cp "$brick_requirements" "$brick_installed_requirements"
          fi
        fi
      fi
    fi
  done
  
  # clean up cache after all brick installs
  uv cache clean
fi

# Pre-provision ALSA wrapped devices
bash /provision-alsa-devices.sh

# Load custom bricks if present
if [ -d "$BASE_DIR/bricks" ]; then
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="$BASE_DIR/bricks"
    else
        export PYTHONPATH="$PYTHONPATH:$BASE_DIR/bricks"
    fi
fi

check_streamlit_ui() {
  grep -q "arduino:streamlit_ui" "$APP_YAML"
}

install_streamlit() {
  if check_streamlit_ui && ! uv pip show streamlit > /dev/null 2>&1; then
    echo "streamlit not found, installing..."
    uv pip install --no-cache-dir --link-mode=copy --compile-bytecode pyarrow==20.0.0 streamlit
  fi
}

case "$1" in
  provision)
    arduino-bricks-list-modules --provision-compose
    ;;
  prepare)
    set -e
    install_streamlit
    if [ "$DEPS_FAILED" -ne 0 ]; then
      echo "======== App preparation failed ====================="
      exit 1
    fi
    # Byte-compile the app sources: turns a syntax error into a prepare
    # failure instead of a crash on the first start.
    if [ -d "$BASE_DIR/bricks" ]; then
      python -m compileall -q "$BASE_DIR/python" "$BASE_DIR/bricks"
    else
      python -m compileall -q "$BASE_DIR/python"
    fi
    ;;
  *)
    install_streamlit
    if check_streamlit_ui; then
      exec streamlit run --server.port 7000 "$PYTHON_SCRIPT"
    fi

    echo "======== App is starting ============================"
    cd $BASE_DIR # Change to the base directory
    exec python "$PYTHON_SCRIPT"
    ;;
esac
