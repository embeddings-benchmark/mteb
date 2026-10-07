import warnings


def warn_if_unsupported_precision(precision: str | None) -> None:
    """Handle embedding quantization options for adapters that return scores."""
    if precision not in {None, "float32"}:
        warnings.warn(
            "Cross-encoder prediction does not support embedding quantization. "
            f"Ignoring precision={precision!r}. Model and scoring dtypes are unchanged.",
            stacklevel=3,
        )
