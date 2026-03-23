"""Test that reference models have results for all public tasks in target benchmarks."""

import logging

import pytest

import mteb
from mteb import ResultCache
from mteb.abstasks import AbsTaskRetrieval

logging.basicConfig(level=logging.INFO)

REFERENCE_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-multilingual-128M",
    "mteb/baseline-bm25s",
]

# Models that can only run retrieval tasks
RETRIEVAL_ONLY_MODELS = {"mteb/baseline-bm25s"}

# Benchmarks excluded from reference model testing.
# All leaderboard benchmarks are tested except these.
# When adding a new benchmark to the leaderboard, ensure reference models
# have results for it, or add it to this exclusion list.
EXCLUDED_BENCHMARKS = {
    # Multimodal benchmarks (image, audio, video)
    "HUME(v1)",
    "JinaVDR",
    "KoViDoRe(v2)",
    "MAEB(beta)",
    "MAEB(beta, audio-only)",
    "MAEB(beta, extended)",
    "MAEB+(beta)",
    "MIEB(eng)",
    "MIEB(Img)",
    "MIEB(lite)",
    "MIEB(Multilingual)",
    "ViDoRe(v1)",
    "ViDoRe(v2)",
    "ViDoRe(v3)",
    "ViDoRe(v3.1)",
    "ViDoRe(v1&v2)",
    # Single-language benchmarks
    "Encodechka",
    "JMTEB-lite(v1)",
    "JMTEB(v2)",
    "MTEB(cmn, v1)",
    "MTEB(deu, v1)",
    "MTEB(fas, v1)",
    "MTEB(fas, v2)",
    "MTEB(fra, v1)",
    "MTEB(jpn, v1)",
    "MTEB(kor, v1)",
    "MTEB(nld, v1)",
    "MTEB(pol, v1)",
    "MTEB(rus, v1)",
    "MTEB(rus, v1.1)",
    "MTEB(spa, v1)",
    "MTEB(tha, v1)",
    "RuSciBench",
    "VN-MTEB (vie, v1)",
    # Specialized/domain-specific benchmarks
    "BEIR",
    "BEIR-NL",
    "BRIGHT",
    "BRIGHT (long)",
    "BRIGHT(v1.1)",
    "BuiltBench(eng)",
    "ChemTEB",
    "ChemTEB(v1.1)",
    "CodeRAG",
    "CoIR",
    "FollowIR",
    "LongEmbed",
    "MINERSBitextMining",
    "MTEB(Code, v1)",
    "MTEB(Law, v1)",
    "MTEB(Medical, v1)",
    "NanoBEIR",
    "R2MED",
    "RAR-b",
    # Older versions of included benchmarks
    "MTEB(eng, v1)",
    "MTEB(Multilingual, v1)",
    # RTEB variants (all covered by RTEB(beta))
    "RTEB(Code, beta)",
    "RTEB(eng, beta)",
    "RTEB(fin, beta)",
    "RTEB(fra, beta)",
    "RTEB(deu, beta)",
    "RTEB(Health, beta)",
    "RTEB(jpn, beta)",
    "RTEB(Law, beta)",
}


def _get_target_benchmarks():
    """Get all leaderboard benchmarks minus excluded ones."""
    all_benchmarks = mteb.get_benchmarks(display_on_leaderboard=True)
    return [b.name for b in all_benchmarks if b.name not in EXCLUDED_BENCHMARKS]


def _is_text_only_task(task):
    """Check if a task uses only text modalities."""
    return task.metadata.modalities == ["text"]


def _get_expected_task_names(benchmark, model_name):
    """Get public task names, filtering by model capabilities."""
    task_names = []
    for task in benchmark:
        if not task.metadata.is_public:
            continue
        if task.is_aggregate:
            task_names.append(task.metadata.name)
            continue
        if not _is_text_only_task(task):
            continue
        if model_name in RETRIEVAL_ONLY_MODELS and not isinstance(
            task, AbsTaskRetrieval
        ):
            continue
        task_names.append(task.metadata.name)
    return task_names


@pytest.fixture(scope="module")
def result_cache():
    cache = ResultCache()
    cache.download_from_remote()
    return cache


TARGET_BENCHMARKS = _get_target_benchmarks()


@pytest.mark.test_reference_models
@pytest.mark.parametrize("benchmark_name", TARGET_BENCHMARKS)
@pytest.mark.parametrize("model_name", REFERENCE_MODELS)
def test_reference_model_coverage(result_cache, benchmark_name, model_name):
    benchmark = mteb.get_benchmark(benchmark_name=benchmark_name)
    expected = _get_expected_task_names(benchmark, model_name)
    results = result_cache.load_results(models=[model_name])
    available = set(results.task_names)
    missing = sorted(set(expected) - available)
    if missing:
        pytest.fail(
            f"'{model_name}' missing {len(missing)}/{len(expected)} tasks "
            f"in '{benchmark_name}':\n" + "\n".join(f"  - {t}" for t in missing)
        )
