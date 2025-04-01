from __future__ import annotations

import difflib
import logging
import warnings

import mteb.benchmarks.benchmarks as benchmark_module
from mteb.benchmarks.benchmarks import (
    C_MTEB,
    FA_MTEB,
    MTEB_DEU,
    MTEB_EN,
    MTEB_ENG_CLASSIC,
    MTEB_EU,
    MTEB_FRA,
    MTEB_INDIC,
    MTEB_JPN,
    MTEB_KOR,
    MTEB_MAIN_RU,
    MTEB_POL,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_MEDICAL,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
    SEB,
    Benchmark,
    MTEB_code,
    MTEB_multilingual,
)

logger = logging.getLogger(__name__)

BENCHMARK_REGISTRY = {
    inst.name: inst
    for nam, inst in benchmark_module.__dict__.items()
    if isinstance(inst, Benchmark)
}

PREVIOUS_BENCHMARK_NAMES = {
    "MTEB(eng)": MTEB_EN.name,
    "MTEB(eng, classic)": MTEB_ENG_CLASSIC.name,
    "MTEB(rus)": MTEB_MAIN_RU.name,
    "MTEB(Retrieval w/Instructions)": MTEB_RETRIEVAL_WITH_INSTRUCTIONS.name,
    "MTEB(law)": MTEB_RETRIEVAL_LAW.name,
    "MTEB(Medical)": MTEB_RETRIEVAL_MEDICAL.name,
    "MTEB(Scandinavian)": SEB.name,
    "MTEB(fra)": MTEB_FRA.name,
    "MTEB(deu)": MTEB_DEU.name,
    "MTEB(kor)": MTEB_KOR.name,
    "MTEB(pol)": MTEB_POL.name,
    "MTEB(code)": MTEB_code.name,
    "MTEB(Multilingual)": MTEB_multilingual.name,
    "MTEB(jpn)": MTEB_JPN.name,
    "MTEB(Indic)": MTEB_INDIC.name,
    "MTEB(Europe)": MTEB_EU.name,
    "MTEB(Chinese)": C_MTEB.name,
    "FaMTEB(fas, beta)": FA_MTEB.name,
}


def get_benchmark(
    benchmark_name: str,
) -> Benchmark:
    if benchmark_name in PREVIOUS_BENCHMARK_NAMES:
        warnings.warn(
            f"Using the previous benchmark name '{benchmark_name}' is deprecated. Please use '{PREVIOUS_BENCHMARK_NAMES[benchmark_name]}' instead.",
            DeprecationWarning,
        )
        benchmark_name = PREVIOUS_BENCHMARK_NAMES[benchmark_name]
    if benchmark_name not in BENCHMARK_REGISTRY:
        close_matches = difflib.get_close_matches(
            benchmark_name, BENCHMARK_REGISTRY.keys()
        )
        if close_matches:
            suggestion = f"KeyError: '{benchmark_name}' not found. Did you mean: {close_matches[0]}?"
        else:
            suggestion = f"KeyError: '{benchmark_name}' not found and no similar keys were found."
        raise KeyError(suggestion)
    return BENCHMARK_REGISTRY[benchmark_name]


def get_benchmarks(
    names: list[str] | None = None, display_on_leaderboard: bool | None = None
) -> list[Benchmark]:
    if names is None:
        names = list(BENCHMARK_REGISTRY.keys())
    benchmarks = [get_benchmark(name) for name in names]
    if display_on_leaderboard is not None:
        benchmarks = [
            b for b in benchmarks if b.display_on_leaderboard is display_on_leaderboard
        ]
    return benchmarks
