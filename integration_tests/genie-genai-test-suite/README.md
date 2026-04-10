# Genie GenAI Test Suite

Integration tests for LLM and VLM functionality using LangChain.

## Test Files

- **test_1_llm_tools.py** - LLM with tool calling capabilities
- **test_2_vlm.py** - Vision Language Model (VLM) tests
- **test_3_vlm_llm.py** - Combined VLM + LLM workflow with enrichment

## Quick Start

### Using the Script

The simplest way to run the tests.

**Windows:**
```batch
run_tests.bat
```

**Linux/Mac:**
```bash
chmod +x run_tests.sh  # First time only
./run_tests.sh
```

The script will:
- Set all required environment variables automatically
- Display the configuration being used
- Run all tests with verbose output

To run a specific test file:

**Windows:**
```batch
run_tests.bat test_1_llm_tools.py
run_tests.bat test_2_vlm.py
run_tests.bat test_3_vlm_llm.py
```

**Linux/Mac:**
```bash
./run_tests.sh test_1_llm_tools.py
./run_tests.sh test_2_vlm.py
./run_tests.sh test_3_vlm_llm.py
```

**To customize settings:** Edit the 6 variables at the top of `run_tests.bat` (Windows) or `run_tests.sh` (Linux/Mac)

### Manual Setup

1. Install dependencies:
   ```bash
   pip install pytest langchain-openai
   ```

2. Set environment variables (modify values as needed):
   
   **Windows:**
   ```batch
   set BASE_URL=http://127.0.0.1:9001/v1
   set TEMPERATURE=0.7
   set VLM_MODEL=qwen3-vl-4b
   set LLM_MODEL=qwen2.5-7b
   set VLM_IMAGES_DIR=VLM-IMAGES
   set OPENAI_API_KEY=xxxx
   ```
   
   **Linux/Mac:**
   ```bash
   export BASE_URL=http://127.0.0.1:9001/v1
   export TEMPERATURE=0.7
   export VLM_MODEL=qwen3-vl-4b
   export LLM_MODEL=qwen2.5-7b
   export VLM_IMAGES_DIR=VLM-IMAGES
   export OPENAI_API_KEY=xxxx
   ```

3. Run tests:
   ```bash
   pytest -v -s
   ```

## Configuration

All configuration is done via environment variables. The test runner scripts (`run_tests.bat` for Windows, `run_tests.sh` for Linux/Mac) use a simplified configuration with just 6 variables that you need to modify. The scripts automatically derive the specific environment variables needed by each test file.

### Main Configuration Variables

- `BASE_URL` - Base URL for both LLM and VLM APIs (default: `http://127.0.0.1:9001/v1`)
- `TEMPERATURE` - Temperature setting for both LLM and VLM (default: `0.7`)
- `VLM_MODEL` - Vision Language Model name (default: `qwen3-vl-4b`)
- `LLM_MODEL` - Large Language Model name (default: `qwen2.5-7b`)
- `VLM_IMAGES_DIR` - Directory containing test images (default: `VLM-IMAGES`)
- `OPENAI_API_KEY` - API key for LangChain OpenAI client (default: `xxxx`)

## Test Images

For VLM tests to run properly, ensure you have test images in the `VLM-IMAGES` directory (or the directory specified by `VLM_IMAGES_DIR`).

Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`

## Test Details

### test_1_llm_tools.py
Tests LLM functionality with tool calling:
- Basic tool definition and execution (`get_current_weather`)
- Streaming and invoke modes
- Multi-turn conversations with tool use
- **Explicit verification that LLM responds after receiving tool results**
- **Assertions ensure final responses are not empty and contain meaningful content**
- Tests for multiple tool calls in a single request

### test_2_vlm.py
Tests Vision Language Model functionality:
- Image processing and analysis
- Streaming and invoke modes
- JSON response parsing
- Per-image testing with dynamic test generation

### test_3_vlm_llm.py
Tests combined VLM + LLM workflow with enrichment:
- VLM analyzes images first (streaming and invoke)
- **LLM enriches VLM output with additional context (streaming, max 30 words)**
- **Ensures LLM is always invoked after VLM completes**
- **Verification that enrichment responses are not empty**
- Comparison testing between streaming and invoke modes
- Tracks and reports timing for both VLM and LLM calls
