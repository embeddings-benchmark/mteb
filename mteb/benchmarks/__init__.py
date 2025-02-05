from __future__ import annotations

from mteb.benchmarks.benchmarks import (
    # BRIGHT,
    # LONG_EMBED,
    # MTEB_DEU,
    # MTEB_EN,
    # MTEB_ENG_CLASSIC,
    # MTEB_EU,
    # MTEB_FRA,
    # MTEB_INDIC,
    # MTEB_JPN,
    # MTEB_KOR,
    # MTEB_MAIN_RU,
    # MTEB_MINERS_BITEXT_MINING,
    # MTEB_POL,
    # MTEB_RETRIEVAL_LAW,
    # MTEB_RETRIEVAL_MEDICAL,
    # MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
    # SEB,
    Benchmark,
    # CoIR,
    # MTEB_code,
    # MTEB_multilingual,
)
from mteb.benchmarks.get_benchmark import (
    BENCHMARK_REGISTRY,
    get_benchmark,
    get_benchmarks,
)

__all__ = [
    "Benchmark",
    "MTEB_EN",
    "MTEB_ENG_CLASSIC",
    "MTEB_MAIN_RU",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_MEDICAL",
    "MTEB_MINERS_BITEXT_MINING",
    "SEB",
    "CoIR",
    "MTEB_FRA",
    "MTEB_DEU",
    "MTEB_KOR",
    "MTEB_POL",
    "MTEB_code",
    "MTEB_multilingual",
    "MTEB_JPN",
    "MTEB_INDIC",
    "MTEB_EU",
    "LONG_EMBED",
    "BRIGHT",
    "BENCHMARK_REGISTRY",
    "get_benchmarks",
    "get_benchmark",
]
