#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# ============================================================================
# Test Runner Script for Genie GenAI Test Suite
# ============================================================================
# This script sets all necessary environment variables and runs the test suite
# Modify the values below to match your environment

echo "========================================"
echo "Genie GenAI Test Suite Runner"
echo "========================================"
echo ""

# ============================================================================
# ENVIRONMENT VARIABLES - Modify these for your setup
# ============================================================================

export BASE_URL=http://127.0.0.1:9001/v1
export TEMPERATURE=0.7
export VLM_MODEL=qwen3-vl-4b
export LLM_MODEL=qwen2.5-3b
export LARGE_LLM_MODEL=qwen2.5-7b
export TIMEOUT=30

# Images directory for VLM tests
export VLM_IMAGES_DIR=VLM-IMAGES

# API Key (required by LangChain OpenAI client)
export OPENAI_API_KEY=xxxx

# ============================================================================
# Display Configuration
# ============================================================================
echo "Configuration:"
echo "  BASE_URL            = $BASE_URL"
echo "  TEMPERATURE         = $TEMPERATURE"
echo "  VLM_MODEL           = $VLM_MODEL"
echo "  LLM_MODEL           = $LLM_MODEL"
echo "  VLM_IMAGES_DIR      = $VLM_IMAGES_DIR"
echo "  OPENAI_API_KEY      = $OPENAI_API_KEY"
echo "  TIMEOUT             = $TIMEOUT"
echo "  LARGE_LLM_MODEL     = $LARGE_LLM_MODEL"
echo ""
echo "========================================"

# ============================================================================
# Run Tests
# ============================================================================

# Check if specific test file is provided as argument
if [ -z "$1" ]; then
    echo "Running all tests..."
    echo ""
    pytest -v -s
else
    echo "Running specific test: $1"
    echo ""
    pytest "$1" -v -s
fi

# ============================================================================
# Capture exit code
# ============================================================================
TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Tests completed successfully!"
else
    echo "Tests failed with exit code: $TEST_EXIT_CODE"
fi
echo "========================================"

exit $TEST_EXIT_CODE
