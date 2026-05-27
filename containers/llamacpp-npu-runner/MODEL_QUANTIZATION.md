# Model Quantization Guide

This guide walks through quantizing a model from raw weights (FP32) to a smaller, optimized format using [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Prerequisites

- **llama.cpp** — You can use the build available inside the container, or download and compile it from the [official repository](https://github.com/ggml-org/llama.cpp).
- **Python 3** with `pip`

## Step 1: Download original model's weights from Hugging Face (optional)

Install the Hugging Face Hub client and download the model weights:

```bash
pip install huggingface_hub
```

```bash
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/gemma-4-E2B-it",
    local_dir="gemma-4-E2B-it"
)
EOF
```

## Step 2: Convert the Model to GGUF (optional)

Before quantizing, the model must first be converted to the GGUF format:

```bash
pip install torch transformer

python convert_hf_to_gguf.py \
    gemma-4-E2B-it \
    --outfile gemma-4-E2B-it-f16.gguf \
    --outtype f16
```

## Step 3: Apply Quantization

Run the quantization tool to produce the final optimized model. You can use pre-converted GGUG BF16 model
or the one previously converted:

```bash
./llama-quantize \
    --pure \
    gemma-4-E2B-it-f16.gguf \
    gemma-4-E2B-it-Q4_0.gguf \
    Q4_0
```

> **Note:** The `--pure` flag forces the backend to strictly apply the requested quantization type to all layers.
> It's possible to change quantization for some specific layes, like embeddings with `--token-embedding-type Q8_0`.
