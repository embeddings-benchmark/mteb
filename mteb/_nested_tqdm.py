from __future__ import annotations

from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def nested_tqdm(parent_pbar):
    """Context manager that makes all tqdm progress bars created within
    nested under the parent progress bar.

    Args:
        parent_pbar: The parent tqdm progress bar

    Usage:
        import time

        # Example usage
        pbar = tqdm(range(10), desc="Main")
        for i in pbar:
            time.sleep(0.1)
            with nested_tqdm(pbar):
                [time.sleep(0.1) for i in tqdm(range(5), desc="Nested")]
    """
    # Get the current position of the parent
    parent_pos = parent_pbar.pos if hasattr(parent_pbar, "pos") else 0

    # Store the original tqdm class
    original_tqdm_init = tqdm.__init__

    # Create a wrapper that adds position parameter
    def wrapped_init(self, *args, **kwargs):
        # Set position to be below parent if not explicitly set
        if "position" not in kwargs:
            kwargs["position"] = parent_pos + 1
        if "leave" not in kwargs:
            kwargs["leave"] = False

        # Auto-add tree prefix to description if not already present
        if (
            "desc" in kwargs
            and kwargs["desc"]
            and not kwargs["desc"].startswith("  └─")
        ):
            kwargs["desc"] = f"  └─ {kwargs['desc']}"
        elif len(args) > 0 and hasattr(args[0], "__iter__"):
            # If desc is positional (rarely used but possible)
            pass

        original_tqdm_init(self, *args, **kwargs)

    # Monkey patch tqdm
    tqdm.__init__ = wrapped_init

    try:
        yield
    finally:
        # Restore original tqdm
        tqdm.__init__ = original_tqdm_init
