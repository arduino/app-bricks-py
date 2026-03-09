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

set BASE_URL=http://192.168.1.140:11434/v1
set TEMPERATURE=0.7
set VLM_MODEL=moondream:latest
set LLM_MODEL=qwen3:4b

REM Images directory for VLM tests
set VLM_IMAGES_DIR=VLM-IMAGES

REM API Key (required by LangChain OpenAI client)
set OPENAI_API_KEY=xxxx

REM ============================================================================
REM Set derived environment variables for the test suite
REM ============================================================================

set OLLAMA_BASE_URL=%BASE_URL%
set OLLAMA_MODEL=%LLM_MODEL%
set OLLAMA_TEMPERATURE=%TEMPERATURE%

set VLM_BASE_URL=%BASE_URL%
set VLM_TEMPERATURE=%TEMPERATURE%

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
