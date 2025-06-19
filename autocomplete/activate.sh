#!/bin/bash
# Activation script for the autocomplete virtual environment

echo "🚀 Activating autocomplete virtual environment..."

# Change to the autocomplete directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Current Python: $(which python)"
echo "📍 Working directory: $(pwd)"
echo ""
echo "Available commands:"
echo "  python src/train_models_optimized.py  - Train the optimized models"
echo "  python api_server.py                  - Start the API server"
echo "  python -m pytest tests/               - Run tests"
echo "  deactivate                           - Exit virtual environment"
echo ""

# Keep the shell open
exec "$SHELL"
