"""Utilities for setting seeds for reproducibility.

Derived from `transformers.trainer_utils.set_seed`.
"""

import random
import sys

import numpy as np


def _set_seed(seed: int) -> tuple[random.Random, np.random.Generator]:
    """Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    If does not set the seed for npu, musa, mlu, hpu, xpu devices. To set the seed for those devices, we recommend using:

    ```python
    from transformers import set_seed
    set_seed(seed)
    ```

    Torch and tensorflow are only seeded if they are already imported, as tasks are created (and seeded) when
    `mteb` is imported. Evaluators seed again when they are created, after the model is loaded.

    Args:
        seed: The seed to set.

    Returns:
        A tuple of (random.Random, np.random.Generator) initialized with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)

    return random.Random(seed), np.random.default_rng(seed)
