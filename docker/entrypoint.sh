#!/bin/bash
set -e

echo "Starting ArchitectAI container process..."

if [ "$1" = "api" ]; then
    echo "Launching ArchitectAI REST API Server on port 8000..."
    exec uvicorn architectai.api.server:app --host 0.0.0.0 --port 8000
elif [ "$1" = "benchmark" ]; then
    echo "Running ArchitectAI experiment benchmark suite..."
    exec python -m experiments.scripts.run_benchmark
else
    exec "$@"
fi
