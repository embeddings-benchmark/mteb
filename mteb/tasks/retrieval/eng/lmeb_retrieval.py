from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal

from huggingface_hub import hf_hub_download

from mteb._evaluators.retrieval_metrics import recall_cap
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

_LMEB_DATASET_REPO = "KaLM-Embedding/LMEB"
_LMEB_DATASET_REVISION = "5b64d963742bd1cdf78e67486dd406bddc846767"
_LMEB_DATASET_DIR_ENV_VARS = ("MTEB_LMEB_DATASET_DIR", "LMEB_DATASET_DIR")
_LMEB_DATE = ("2026-03-19", "2026-03-19")
_LMEB_CITATION = r"""
@misc{zhao2026lmeb,
  archiveprefix = {arXiv},
  author = {Zhao, Xinping and Hu, Xinshuo and Xu, Jiaxin and Tang, Danyu and Zhang, Xin and Zhou, Mengjia and Zhong, Yan and Zhou, Yao and Shan, Zifei and Zhang, Meishan and Hu, Baotian and Zhang, Min},
  eprint = {2603.12572},
  primaryclass = {cs.CL},
  title = {LMEB: Long-horizon Memory Embedding Benchmark},
  url = {https://arxiv.org/abs/2603.12572},
  year = {2026},
}
"""

_DIALOGUE_DOMAINS = ["Social", "Spoken"]
_DOCUMENT_DOMAINS = ["Written"]
_PROCEDURAL_DOMAINS = ["Programming", "Web"]

_EPBENCH_SCENARIOS = [
    "default_claude_long",
    "default_claude_short",
    "default_claude_very_long",
    "default_gpt4o_long",
    "default_gpt4o_short",
    "sci_fi_claude_long",
    "sci_fi_claude_short",
    "world_news_claude_long",
    "world_news_claude_short",
]
_EPBENCH_QUERY_TYPES = [
    "Entities",
    "Event_contents",
    "Full_event_details",
    "Other_entities",
    "Spaces",
    "Times",
]
_KNOWMEBENCH_SCENARIOS = [
    "event_driven",
    "flashback_intensive",
    "psychological_depth",
]
_KNOWMEBENCH_QUERY_TYPES = [
    "adversarial_abstention",
    "information_extraction",
    "mind-body_interaction",
    "mnestic_trigger_analysis",
    "temporal_reasoning",
]
_MEMGOVERN_REPOS = [
    "Azure_azure-sdk-for-python",
    "ClusterHQ_flocker",
    "DataDog_integrations-core",
    "Microsoft_TypeScript",
    "PokemonGoF_PokemonGo-Bot",
    "Qiskit_qiskit-terra",
    "StackStorm_st2",
    "apache_airflow",
    "apache_incubator-airflow",
    "certbot_certbot",
    "dask_dask",
    "datalad_datalad",
    "django_django",
    "dmwm_WMCore",
    "encode_django-rest-framework",
    "facebook_react",
    "great-expectations_great_expectations",
    "home-assistant_core",
    "home-assistant_home-assistant",
    "huggingface_transformers",
    "kubernetes_kubernetes",
    "mesonbuild_meson",
    "microsoft_vscode",
    "mne-tools_mne-python",
    "moby_moby",
    "napari_napari",
    "numpy_numpy",
    "optuna_optuna",
    "pandas-dev_pandas",
    "pydata_xarray",
    "pypa_pip",
    "pytest-dev_pytest",
    "pytorch_pytorch",
    "raiden-network_raiden",
    "rust-lang_rust",
    "saltstack_salt",
    "scikit-learn_scikit-learn",
    "scipy_scipy",
    "scrapy_scrapy",
    "spack_spack",
    "sphinx-doc_sphinx",
    "spotify_luigi",
    "spring-projects_spring-framework",
    "spyder-ide_spyder",
    "sympy_sympy",
    "tensorflow_tensorflow",
    "webpack_webpack",
    "xonsh_xonsh",
]
_REME_VARIANTS = {
    "appworld_qwen3_8b": ["generalized_query"],
    "appworld_qwen3_14b": ["generalized_query"],
    "appworld_qwen3_32b": ["generalized_query"],
    "bfcl_qwen3_8b": ["generalized_query", "task_query"],
    "bfcl_qwen3_14b": ["generalized_query", "task_query"],
    "bfcl_qwen3_32b": ["generalized_query", "task_query"],
}


def _make_eval_langs(subsets: list[str]) -> dict[str, list[str]]:
    return {subset: ["eng-Latn"] for subset in subsets}


@dataclass(frozen=True)
class _LMEBTaskConfig:
    description: str
    relative_path: str
    eval_langs: dict[str, list[str]]
    reference: str
    license: str
    domains: list[str]
    task_subtypes: list[str]
    query_location: Literal["task", "subset"] = "subset"
    corpus_location: Literal["task", "subset", "subset_parent"] = "subset"
    candidate_location: Literal["task", "subset"] | None = None
    candidate_query_strategy: Literal["prefix2", "identity"] | None = None


_LMEB_TASK_CONFIGS = {
    "EPBench": _LMEBTaskConfig(
        description=(
            "LMEB episodic-memory retrieval task based on EPBench, evaluating "
            "event-centric retrieval across generator, context-length, and query-type variants."
        ),
        relative_path="Episodic/EPBench",
        eval_langs=_make_eval_langs(
            [
                f"{scenario}/{query_type}"
                for scenario in _EPBENCH_SCENARIOS
                for query_type in _EPBENCH_QUERY_TYPES
            ]
        ),
        reference="https://doi.org/10.6084/m9.figshare.28244480",
        license="mit",
        domains=["Fiction", "News", "Written"],
        task_subtypes=["Question answering"],
        corpus_location="subset_parent",
    ),
    "KnowMeBench": _LMEBTaskConfig(
        description=(
            "LMEB episodic-memory retrieval task based on KnowMeBench, focused on "
            "person understanding, temporal reasoning, and memory-triggered retrieval."
        ),
        relative_path="Episodic/KnowMeBench",
        eval_langs=_make_eval_langs(
            [
                f"{scenario}/{query_type}"
                for scenario in _KNOWMEBENCH_SCENARIOS
                for query_type in _KNOWMEBENCH_QUERY_TYPES
            ]
        ),
        reference="https://github.com/QuantaAlpha/KnowMeBench/tree/main/KnowmeBench",
        license="apache-2.0",
        domains=["Fiction", "Written"],
        task_subtypes=["Question answering"],
        corpus_location="subset_parent",
    ),
    "LoCoMo": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on LoCoMo, covering single-hop, "
            "multi-hop, temporal, open-domain, and adversarial long-conversation questions."
        ),
        relative_path="Dialogue/LoCoMo",
        eval_langs=_make_eval_langs(
            [
                "single_hop",
                "multi_hop",
                "temporal_reasoning",
                "open_domain",
                "adversarial",
            ]
        ),
        reference="https://github.com/snap-research/locomo/tree/main/data",
        license="cc-by-nc-4.0",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="prefix2",
    ),
    "LongMemEval": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on LongMemEval, evaluating "
            "single-session, multi-session, preference, knowledge-update, and temporal queries."
        ),
        relative_path="Dialogue/LongMemEval",
        eval_langs=_make_eval_langs(
            [
                "knowledge_update",
                "multi_session",
                "single_session_assistant",
                "single_session_preference",
                "single_session_user",
                "temporal_reasoning",
            ]
        ),
        reference="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        license="mit",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="prefix2",
    ),
    "REALTALK": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on REALTALK, targeting commonsense, "
            "multi-hop, and temporal reasoning over realistic conversations."
        ),
        relative_path="Dialogue/REALTALK",
        eval_langs=_make_eval_langs(["commonsense", "multi_hop", "temporal_reasoning"]),
        reference="https://github.com/danny911kr/REALTALK/tree/main/data",
        license="not specified",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="prefix2",
    ),
    "TMD": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on the Temporal Memory Dataset, "
            "evaluating date, session, and relative-time retrieval over dialogues."
        ),
        relative_path="Dialogue/TMD",
        eval_langs=_make_eval_langs(
            [
                "content_time_qs",
                "date_span_time_qs",
                "dates_time_qs",
                "day_span_time_qs",
                "earlier_today_time_qs",
                "last_named_day_time_qs",
                "month_time_qs",
                "rel_day_time_qs",
                "rel_month_time_qs",
                "rel_session_time_qs",
                "session_span_time_qs",
                "session_time_qs",
            ]
        ),
        reference="https://github.com/Zyphra/TemporalMemoryDataset",
        license="not specified",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="prefix2",
    ),
    "MemBench": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on MemBench, measuring retrieval "
            "for preference, emotion, recommendation, update, and multi-hop memory cues."
        ),
        relative_path="Dialogue/MemBench",
        eval_langs=_make_eval_langs(
            [
                "aggregative",
                "comparative",
                "emotion",
                "knowledge_updating",
                "multi_hop",
                "multi_session_assistant",
                "post_processing",
                "preference",
                "single_hop",
                "single_session_assistant",
            ]
        ),
        reference="https://github.com/import-myself/Membench/tree/main/MemData",
        license="mit",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        candidate_location="subset",
        candidate_query_strategy="prefix2",
    ),
    "ConvoMem": _LMEBTaskConfig(
        description=(
            "LMEB dialogue-memory retrieval task based on ConvoMem, centered on evidence "
            "retrieval for facts, preferences, abstention, and changing user state."
        ),
        relative_path="Dialogue/ConvoMem",
        eval_langs=_make_eval_langs(
            [
                "abstention_evidence",
                "assistant_facts_evidence",
                "changing_evidence",
                "implicit_connection_evidence",
                "preference_evidence",
                "user_evidence",
            ]
        ),
        reference="https://huggingface.co/datasets/Salesforce/ConvoMem",
        license="cc-by-nc-4.0",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        candidate_location="subset",
        candidate_query_strategy="prefix2",
    ),
    "QASPER": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on QASPER, retrieving evidence passages "
            "from research papers for information-seeking questions."
        ),
        relative_path="Semantic/QASPER",
        eval_langs=_make_eval_langs(["QASPER"]),
        reference="https://huggingface.co/datasets/allenai/qasper",
        license="cc-by-nc-4.0",
        domains=["Academic", "Written"],
        task_subtypes=["Reading Comprehension"],
        query_location="task",
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "NovelQA": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on NovelQA, retrieving passages from long-form "
            "fiction for character, plot, setting, relation, and temporal questions."
        ),
        relative_path="Semantic/NovelQA",
        eval_langs=_make_eval_langs(
            ["Character", "Meaning", "Plot", "Relation", "Setting", "Span", "Times"]
        ),
        reference="https://huggingface.co/datasets/NovelQA/NovelQA",
        license="not specified",
        domains=["Fiction", "Written"],
        task_subtypes=["Reading Comprehension"],
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "PeerQA": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on PeerQA, retrieving evidence from peer-review "
            "material and scholarly writing to answer questions."
        ),
        relative_path="Semantic/PeerQA",
        eval_langs=_make_eval_langs(["PeerQA"]),
        reference="https://huggingface.co/datasets/UKPLab/PeerQA",
        license="cc-by-nc-sa-4.0",
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        query_location="task",
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "CovidQA": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on COVID-QA, retrieving relevant biomedical "
            "evidence for COVID-19 questions."
        ),
        relative_path="Semantic/Covid-QA",
        eval_langs=_make_eval_langs(["Covid-QA"]),
        reference="https://huggingface.co/datasets/illuin-conteb/covid-qa",
        license="apache-2.0",
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Question answering"],
        query_location="task",
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "ESGReports": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on ESG-Reports, retrieving long-form report "
            "passages relevant to environmental, social, and governance questions."
        ),
        relative_path="Semantic/ESG-Reports",
        eval_langs=_make_eval_langs(["ESG-Reports"]),
        reference="https://huggingface.co/datasets/illuin-conteb/esg-reports",
        license="not specified",
        domains=["Financial", "Written"],
        task_subtypes=["Question answering"],
        query_location="task",
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "MLDR": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on MLDR, focused on long-document retrieval "
            "for question answering in English."
        ),
        relative_path="Semantic/MLDR",
        eval_langs=_make_eval_langs(["MLDR"]),
        reference="https://huggingface.co/datasets/illuin-conteb/mldr-conteb-eval",
        license="mit",
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        query_location="task",
        corpus_location="task",
        candidate_location="task",
        candidate_query_strategy="identity",
    ),
    "LooGLE": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on LooGLE, evaluating long- and short-dependency "
            "question answering over long documents."
        ),
        relative_path="Semantic/LooGLE",
        eval_langs=_make_eval_langs(["LongDepQA", "ShortDepQA"]),
        reference="https://huggingface.co/datasets/bigai-nlco/LooGLE",
        license="cc-by-sa-4.0",
        domains=["Web", "Written"],
        task_subtypes=["Reading Comprehension"],
        candidate_location="subset",
        candidate_query_strategy="identity",
    ),
    "LMEB_SciFact": _LMEBTaskConfig(
        description=(
            "LMEB semantic retrieval task based on SciFact, retrieving scientific evidence "
            "passages for claim verification."
        ),
        relative_path="Semantic/SciFact",
        eval_langs=_make_eval_langs(["LMEB_SciFact"]),
        reference="https://huggingface.co/datasets/allenai/scifact",
        license="cc-by-nc-3.0",
        domains=["Academic", "Written"],
        task_subtypes=["Claim verification"],
        query_location="task",
        corpus_location="task",
    ),
    "Gorilla": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on Gorilla, retrieving model and API documentation "
            "for tool-use and code-generation requests."
        ),
        relative_path="Procedural/Gorilla",
        eval_langs=_make_eval_langs(
            ["gorilla_huggingface", "gorilla_pytorch", "gorilla_tensor"]
        ),
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
    ),
    "ToolBench": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on ToolBench, retrieving API documentation "
            "relevant to a tool-use query."
        ),
        relative_path="Procedural/ToolBench",
        eval_langs=_make_eval_langs(["ToolBench"]),
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        query_location="task",
        corpus_location="task",
    ),
    "ReMe": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on ReMe, retrieving past successful experiences "
            "and procedures for agentic execution."
        ),
        relative_path="Procedural/ReMe",
        eval_langs=_make_eval_langs(
            [
                f"{variant}/{subset}"
                for variant, subsets in _REME_VARIANTS.items()
                for subset in subsets
            ]
        ),
        reference="https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        corpus_location="subset_parent",
    ),
    "Proced_mem_bench": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on Proced_mem_bench, retrieving useful "
            "multi-step procedures for downstream user queries."
        ),
        relative_path="Procedural/Proced_mem_bench",
        eval_langs=_make_eval_langs(["easy", "medium", "hard"]),
        reference="https://github.com/qpiai/Proced_mem_bench/tree/main/procedural_memory_benchmark",
        license="apache-2.0",
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        corpus_location="task",
    ),
    "MemGovern": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on MemGovern, retrieving governed human "
            "experiences from software-engineering repositories."
        ),
        relative_path="Procedural/MemGovern",
        eval_langs=_make_eval_langs(_MEMGOVERN_REPOS),
        reference="https://github.com/QuantaAlpha/MemGovern/blob/main/data",
        license="mit",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
    ),
    "DeepPlanning": _LMEBTaskConfig(
        description=(
            "LMEB procedural retrieval task based on DeepPlanning, retrieving shopping items "
            "needed to support long-horizon planning queries."
        ),
        relative_path="Procedural/DeepPlanning",
        eval_langs=_make_eval_langs(
            ["shopping_level1", "shopping_level2", "shopping_level3"]
        ),
        reference="https://huggingface.co/datasets/Qwen/DeepPlanning",
        license="apache-2.0",
        domains=["E-commerce", "Written"],
        task_subtypes=["Question answering"],
        candidate_location="subset",
        candidate_query_strategy="identity",
    ),
}


def _get_local_dataset_root(
    dataset_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    if dataset_dir is not None:
        root = Path(dataset_dir).expanduser()
        return root if root.exists() else None

    for env_var in _LMEB_DATASET_DIR_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            root = Path(env_value).expanduser()
            if root.exists():
                return root
            logger.warning(
                "LMEB local dataset directory from %s does not exist: %s",
                env_var,
                root,
            )
            return None
    return None


def _download_dataset_file(
    relative_path: PurePosixPath,
    cache_dir: str | None = None,
    dataset_dir: str | os.PathLike[str] | None = None,
) -> Path:
    local_root = _get_local_dataset_root(dataset_dir)
    if local_root is not None:
        local_path = local_root / Path(relative_path.as_posix())
        if local_path.exists():
            return local_path
        logger.warning(
            "LMEB local dataset file not found at %s, falling back to Hugging Face Hub.",
            local_path,
        )

    return Path(
        hf_hub_download(
            repo_id=_LMEB_DATASET_REPO,
            repo_type="dataset",
            filename=relative_path.as_posix(),
            revision=_LMEB_DATASET_REVISION,
            cache_dir=cache_dir,
        )
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, document_id, score = row[:3]
            qrels.setdefault(str(query_id), {})[str(document_id)] = int(score)
    return qrels


def _average_optional(values: list[float | None]) -> float:
    valid = [value for value in values if value is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


class LMEBRetrievalTask(AbsTaskRetrieval):
    query_id_field = "id"
    query_text_field = "text"
    corpus_id_field = "id"
    corpus_title_field = "title"
    corpus_text_field = "text"
    k_values: ClassVar[list[int]] = [1, 5, 10, 25, 50]
    _top_k = max(k_values)
    _task_config: _LMEBTaskConfig

    def _resolve_directory(
        self,
        hf_subset: str,
        location: Literal["task", "subset", "subset_parent"],
    ) -> PurePosixPath:
        base_path = PurePosixPath(self._task_config.relative_path)
        if location == "task":
            return base_path
        if location == "subset":
            return base_path / hf_subset
        return base_path / hf_subset.rsplit("/", maxsplit=1)[0]

    def _resolve_file(
        self,
        hf_subset: str,
        filename: str,
        location: Literal["task", "subset", "subset_parent"],
    ) -> PurePosixPath:
        return self._resolve_directory(hf_subset, location) / filename

    def _build_top_ranked(
        self,
        hf_subset: str,
        query_ids: list[str],
        corpus_ids: list[str],
        *,
        cache_dir: str | None = None,
        dataset_dir: str | os.PathLike[str] | None = None,
    ) -> dict[str, list[str]] | None:
        if self._task_config.candidate_location is None:
            return None

        candidates_path = _download_dataset_file(
            self._resolve_file(
                hf_subset,
                "candidates.jsonl",
                self._task_config.candidate_location,
            ),
            cache_dir=cache_dir,
            dataset_dir=dataset_dir,
        )
        candidate_rows = _read_jsonl(candidates_path)
        candidates_by_scene = {
            str(row["scene_id"]): [str(doc_id) for doc_id in row["candidate_doc_ids"]]
            for row in candidate_rows
        }

        corpus_id_set = set(corpus_ids)
        top_ranked: dict[str, list[str]] = {}
        dropped_candidate_ids = 0
        empty_candidate_queries = 0
        for query_id in query_ids:
            if self._task_config.candidate_query_strategy == "identity":
                scene_id = query_id
            else:
                scene_id = "_".join(query_id.split("_")[:2])

            ranked_ids = candidates_by_scene.get(scene_id)
            if ranked_ids is None:
                top_ranked[query_id] = corpus_ids
                continue

            filtered_ranked_ids: list[str] = []
            seen_doc_ids: set[str] = set()
            for doc_id in ranked_ids:
                if doc_id not in corpus_id_set:
                    dropped_candidate_ids += 1
                    continue
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                filtered_ranked_ids.append(doc_id)

            if filtered_ranked_ids:
                top_ranked[query_id] = filtered_ranked_ids
            else:
                empty_candidate_queries += 1
                top_ranked[query_id] = corpus_ids

        if dropped_candidate_ids > 0 or empty_candidate_queries > 0:
            logger.warning(
                "LMEB %s/%s dropped %d candidate doc ids not present in the corpus; "
                "fell back to the full corpus for %d queries.",
                self.metadata.name,
                hf_subset,
                dropped_candidate_ids,
                empty_candidate_queries,
            )

        return top_ranked

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        del num_proc
        if self.data_loaded:
            return

        cache_dir = kwargs.get("cache_dir")
        dataset_dir = kwargs.get("dataset_dir")

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}
        self.top_ranked = {}

        for hf_subset in self.metadata.hf_subsets:
            self.corpus[hf_subset] = {}
            self.queries[hf_subset] = {}
            self.relevant_docs[hf_subset] = {}
            self.top_ranked[hf_subset] = {}

            queries_path = _download_dataset_file(
                self._resolve_file(
                    hf_subset,
                    "queries.jsonl",
                    self._task_config.query_location,
                ),
                cache_dir=cache_dir,
                dataset_dir=dataset_dir,
            )
            corpus_path = _download_dataset_file(
                self._resolve_file(
                    hf_subset,
                    "corpus.jsonl",
                    self._task_config.corpus_location,
                ),
                cache_dir=cache_dir,
                dataset_dir=dataset_dir,
            )
            qrels_path = _download_dataset_file(
                self._resolve_file(
                    hf_subset,
                    "qrels.tsv",
                    self._task_config.query_location,
                ),
                cache_dir=cache_dir,
                dataset_dir=dataset_dir,
            )

            queries = {
                str(row[self.query_id_field]): row[self.query_text_field]
                for row in _read_jsonl(queries_path)
            }
            corpus = {
                str(row[self.corpus_id_field]): {
                    "title": row.get(self.corpus_title_field, "") or "",
                    "text": row.get(self.corpus_text_field, "") or "",
                }
                for row in _read_jsonl(corpus_path)
            }
            qrels = _read_qrels(qrels_path)
            top_ranked = self._build_top_ranked(
                hf_subset,
                list(queries.keys()),
                list(corpus.keys()),
                cache_dir=cache_dir,
                dataset_dir=dataset_dir,
            )

            for split in self.eval_splits:
                self.corpus[hf_subset][split] = corpus
                self.queries[hf_subset][split] = queries
                self.relevant_docs[hf_subset][split] = qrels
                self.top_ranked[hf_subset][split] = top_ranked

        self.data_loaded = True

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        del scores, hf_split, hf_subset
        capped_recall = recall_cap(qrels, results, self.k_values)
        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


def _create_lmeb_task(name: str, config: _LMEBTaskConfig) -> type[LMEBRetrievalTask]:
    metadata = TaskMetadata(
        name=name,
        dataset={
            "path": _LMEB_DATASET_REPO,
            "revision": _LMEB_DATASET_REVISION,
        },
        description=config.description,
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        reference=config.reference,
        eval_splits=["test"],
        eval_langs=config.eval_langs,
        main_score="ndcg_at_10",
        date=_LMEB_DATE,
        domains=config.domains,
        task_subtypes=config.task_subtypes,
        license=config.license,
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_LMEB_CITATION,
    )
    return type(
        name,
        (LMEBRetrievalTask,),
        {
            "__module__": __name__,
            "_task_config": config,
            "metadata": metadata,
        },
    )


for _task_name, _task_config in _LMEB_TASK_CONFIGS.items():
    globals()[_task_name] = _create_lmeb_task(_task_name, _task_config)


__all__ = list(_LMEB_TASK_CONFIGS)
