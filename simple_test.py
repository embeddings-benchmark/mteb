from __future__ import annotations

import os
import sys

import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Try to import AudioPairClassificationEvaluator directly
try:
    # Add the parent directory to the path
    sys.path.insert(0, os.path.abspath("."))

    print("Successfully imported AudioPairClassificationEvaluator")

    # Test the logger modification
    import logging

    logger = logging.getLogger(
        "mteb.evaluation.evaluators.Audio.AudioPairClassificationEvaluator"
    )
    logger.setLevel(logging.WARN)
    print("Logger configured successfully")

except Exception as e:
    print(f"Import error: {e}")
    import traceback

    traceback.print_exc()
