#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

if [ -z "$PYTHONUNBUFFERED" ]; then
  export PYTHONUNBUFFERED=1
fi

BASE_DIR="/app"
CACHE_DIR="$BASE_DIR/.cache"
APP_YAML="$BASE_DIR/app.yaml"
PYTHON_SCRIPT="$BASE_DIR/python/main.py"
REQUIREMENTS_FILE="$BASE_DIR/python/requirements.txt"
PYTHON_LIBS_DIR="$BASE_DIR/python-libraries"
INSTALLED_REQUIREMENTS_FILE="$CACHE_DIR/installed_requirements.txt"

export UV_CACHE_DIR="$CACHE_DIR/uv"

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

echo "Activating python virtual environment"
. "$CACHE_DIR/.venv/bin/activate"

if [ -d "$PYTHON_LIBS_DIR" ]; then
  echo "Installing Python libraries from $PYTHON_LIBS_DIR"
  # Iterate over each .whl file in the directory
  for whl_file in "$PYTHON_LIBS_DIR"/*.whl; do
    if [ -f "$whl_file" ]; then
      uv pip install "$whl_file"
      mv "$whl_file" "$whl_file".installed
    fi
  done
fi

if [ -f "$REQUIREMENTS_FILE" ]; then
  INSTALL_DEPS=1
  REQUIREMENTS_LINES="$(cat $REQUIREMENTS_FILE | grep -c '[^[:space:]]')"
  if [ -f "$INSTALLED_REQUIREMENTS_FILE" ]; then
    if cmp -s "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"; then
        echo "Requirements already installed."
    else
        INSTALL_DEPS=0
    fi
  fi
  if [ "$INSTALL_DEPS" -gt 0 ]; then
    if [ "$REQUIREMENTS_LINES" -ne 0 ]; then
      uv pip install -r "$REQUIREMENTS_FILE"
    fi
  fi
  # clean up cache
  uv cache clean
  cp "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"
fi

if [ "$1" = "provision" ]; then
  arduino-bricks-list-modules --provision-compose
else
  if grep -q "arduino:streamlit_ui" "$APP_YAML"; then
    exec streamlit run --server.port 7000 "$PYTHON_SCRIPT"
  else
    cd $BASE_DIR # Change to the base directory
    exec python "$PYTHON_SCRIPT"
  fi
fi
