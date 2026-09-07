#!/usr/bin/env bash
# Verifies the RAMDeck node agent's RPC connection by downloading a tiny test model
# and sending a tensor offload request to the node agent.
#
# Requirements:
# 1. The RAMDeck Node Agent must be running (python -m daemon.ramdeck.node_agent)
# 2. You must have `llama-cli` installed in your PATH.

set -e

MODEL_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"
MODEL_FILE="stories15M-q4_0.gguf"
EXPECTED_MIN_SIZE=5000000 # ~5MB

echo "Checking for llama-cli..."
if ! command -v llama-cli &> /dev/null; then
    echo "ERROR: llama-cli is not installed or not in PATH."
    echo "Please install llama.cpp before running this verification."
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading TinyStories-15M (fast sanity check model, ~9MB)..."
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
else
    echo "Model $MODEL_FILE already exists."
fi

# Sanity check the download size
FILE_SIZE=$(wc -c < "$MODEL_FILE" | tr -d ' ')
if [ "$FILE_SIZE" -lt "$EXPECTED_MIN_SIZE" ]; then
    echo "ERROR: Downloaded file is too small ($FILE_SIZE bytes)."
    echo "The download may have failed or corrupted. Please delete $MODEL_FILE and try again."
    exit 1
fi

echo "Download verified successfully ($FILE_SIZE bytes)."

echo ""
echo "============================================================"
echo "Sending inference task to RAMDeck Node Agent via RPC..."
echo "NOTE: TinyStories outputs will likely be gibberish."
echo "The goal is verifying the RPC pipe processes the tensors!"
echo "============================================================"
echo ""

# Point llama-cli at the local node agent's RPC port (50052)
llama-cli \
    -m "$MODEL_FILE" \
    --rpc 127.0.0.1:50052 \
    -n 32 \
    -p "Once upon a time" \
    --temp 0.1

echo ""
echo "============================================================"
echo "Verification Complete! If you saw generated text above,"
echo "your RAMDeck Node Agent is successfully offloading tensors."
echo "============================================================"
