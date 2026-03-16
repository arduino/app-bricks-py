REM SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
REM
REM SPDX-License-Identifier: MPL-2.0

@echo off
REM ============================================================================
REM Test Runner Script for Genie GenAI Test Suite
REM ============================================================================
REM This script sets all necessary environment variables and runs the test suite
REM Modify the values below to match your environment

echo ========================================
echo Genie GenAI Test Suite Runner
echo ========================================
echo.

REM ============================================================================
REM ENVIRONMENT VARIABLES - Modify these for your setup
REM ============================================================================

set BASE_URL=http://127.0.0.1:9001/v1
set TEMPERATURE=0.7
set VLM_MODEL=qwen3-vl-4b
set LLM_MODEL=qwen2.5-3b
set LARGE_LLM_MODEL=qwen2.5-7b
set TIMEOUT=30

REM Images directory for VLM tests
set VLM_IMAGES_DIR=VLM-IMAGES

REM API Key (required by LangChain OpenAI client)
set OPENAI_API_KEY=xxxx

REM ============================================================================
REM Display Configuration
REM ============================================================================
echo Configuration:
echo   BASE_URL            = %BASE_URL%
echo   TEMPERATURE         = %TEMPERATURE%
echo   VLM_MODEL           = %VLM_MODEL%
echo   LLM_MODEL           = %LLM_MODEL%
echo   VLM_IMAGES_DIR      = %VLM_IMAGES_DIR%
echo   OPENAI_API_KEY      = %OPENAI_API_KEY%
echo   TIMEOUT             = %TIMEOUT%
echo   LARGE_LLM_MODEL     = %LARGE_LLM_MODEL%
echo.
echo ========================================

REM ============================================================================
REM Run Tests
REM ============================================================================

REM Check if specific test file is provided as argument
if "%1"=="" (
    echo Running all tests...
    echo.
    pytest -v -s
) else (
    echo Running specific test: %1
    echo.
    pytest %1 -v -s
)

REM ============================================================================
REM Capture exit code
REM ============================================================================
set TEST_EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %TEST_EXIT_CODE% equ 0 (
    echo Tests completed successfully!
) else (
    echo Tests failed with exit code: %TEST_EXIT_CODE%
)
echo ========================================

exit /b %TEST_EXIT_CODE%
