"""Utilities for setting seeds for reproducibility.

Derived from `transformers.trainer_utils.set_seed`. It assumes torch is installed.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> tuple[random.Random, np.random.Generator]:
    """Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    If does not set the seed for npu, musa, mlu, hpu, xpu devices. To set the seed for those devices, we recommend using:

    ```python
    from transformers import set_seed
    set_seed(seed)
    ```

    Args:
        seed: The seed to set.

    Returns:
        A tuple of (random.Random, np.random.Generator) initialized with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass  # not installed

    return random.Random(seed), np.random.default_rng(seed)
