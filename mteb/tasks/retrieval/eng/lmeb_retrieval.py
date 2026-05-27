from __future__ import annotations

import logging

from mteb._evaluators.retrieval_metrics import recall_cap
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

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
_LMEB_K_VALUES = (1, 3, 5, 10, 20, 25, 50, 100, 1000)

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

_common_metadata = dict(
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    main_score="ndcg_at_10",
    annotations_creators="derived",
    dialect=[],
    sample_creation="found",
    date=_LMEB_DATE,
    bibtex_citation=_LMEB_CITATION,
)


def _average_optional(values: list[float | None]) -> float:
    valid = [value for value in values if value is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


class EPBench(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="EPBench",
        dataset={
            "path": "mteb/EPBench",
            "revision": "c0c0877aab7085655b446b2adcb86a25bbf14459",
        },
        description=(
            "LMEB episodic-memory retrieval task based on EPBench. Queries ask "
            "about entities, event contents, spaces, times, and full event details "
            "in synthetic event narratives; the corpus contains generated event "
            "memories, and the goal is to retrieve the relevant event records."
        ),
        eval_langs={
            f"{scenario}__{query_type}": ["eng-Latn"]
            for scenario in _EPBENCH_SCENARIOS
            for query_type in _EPBENCH_QUERY_TYPES
        },
        reference="https://doi.org/10.6084/m9.figshare.28244480",
        license="mit",
        domains=["Fiction", "News", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class DeepPlanning(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="DeepPlanning",
        dataset={
            "path": "mteb/DeepPlanning",
            "revision": "802c327a8b88bf36b17a39d12df7e233efe8d079",
        },
        description=(
            "LMEB procedural retrieval task based on DeepPlanning. Queries "
            "describe long-horizon shopping and planning needs; the corpus "
            "contains shopping item memories, and the goal is to retrieve the "
            "items needed to support the plan."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in ["shopping_level1", "shopping_level2", "shopping_level3"]
        },
        reference="https://huggingface.co/datasets/Qwen/DeepPlanning",
        license="apache-2.0",
        domains=["E-commerce", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class MemGovern(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="MemGovern",
        dataset={
            "path": "mteb/MemGovern",
            "revision": "2dc71218b15aa0e987ecc2f35ece6a1bc769e19c",
        },
        description=(
            "LMEB procedural retrieval task based on MemGovern. Queries ask "
            "software-engineering governance questions over repository histories; "
            "the corpus contains human experience records from software projects, "
            "and the goal is to retrieve the relevant governed experiences."
        ),
        eval_langs={repo: ["eng-Latn"] for repo in _MEMGOVERN_REPOS},
        reference="https://github.com/QuantaAlpha/MemGovern/blob/main/data",
        license="mit",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class ProceduralMemBench(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="ProceduralMemBench",
        dataset={
            "path": "mteb/Proced_mem_bench",
            "revision": "5e4517cf2f768957b83917ef1961a5bd880675cd",
        },
        description=(
            "LMEB procedural retrieval task based on ProceduralMemBench. Queries "
            "describe downstream user goals that require procedures; the corpus "
            "contains multi-step procedural memories, and the goal is to retrieve "
            "the procedure useful for the query."
        ),
        eval_langs={subset: ["eng-Latn"] for subset in ["easy", "medium", "hard"]},
        reference="https://github.com/qpiai/Proced_mem_bench/tree/main/procedural_memory_benchmark",
        license="apache-2.0",
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class ReMe(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="ReMe",
        dataset={
            "path": "mteb/ReMe",
            "revision": "7bbb4456dade71ab22c9373c9a2f3504cfdccc18",
        },
        description=(
            "LMEB procedural retrieval task based on ReMe. Queries are generalized "
            "or task-specific agent execution requests; the corpus contains past "
            "successful agent experiences and procedures, and the goal is to "
            "retrieve experience that can guide the next execution."
        ),
        eval_langs={
            f"{variant}__{subset}": ["eng-Latn"]
            for variant, subsets in _REME_VARIANTS.items()
            for subset in subsets
        },
        reference="https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class ToolBench(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="ToolBench",
        dataset={
            "path": "mteb/ToolBench",
            "revision": "6f868e5b4c46a9f04c89f7179ddc7c8f162c1068",
        },
        description=(
            "LMEB procedural retrieval task based on ToolBench. Queries describe "
            "tool-use and API needs; the corpus contains API and tool documentation, "
            "and the goal is to retrieve documentation relevant to the requested tool."
        ),
        eval_langs={"ToolBench": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class Gorilla(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="Gorilla",
        dataset={
            "path": "mteb/Gorilla",
            "revision": "ed4eb0d3e13cd43fd4edff0f150de3b501511244",
        },
        description=(
            "LMEB procedural retrieval task based on Gorilla. Queries are model, "
            "API, and code-generation requests; the corpus contains Hugging Face, "
            "PyTorch, and TensorFlow documentation, and the goal is to retrieve "
            "the documentation needed for correct API use."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in ["gorilla_huggingface", "gorilla_pytorch", "gorilla_tensor"]
        },
        reference="https://huggingface.co/datasets/mangopy/ToolRet-Queries",
        license="apache-2.0",
        domains=_PROCEDURAL_DOMAINS,
        task_subtypes=["Code retrieval"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class LMEBSciFact(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="LMEB_SciFact",
        dataset={
            "path": "mteb/LMEB_SciFact",
            "revision": "432cf48c879c2de792720d5c0af6900312ebe7b6",
        },
        description=(
            "LMEB-formatted SciFact retrieval task. Queries are scientific claims "
            "from LMEB; the corpus contains scientific evidence passages, and the "
            "goal is to retrieve evidence for claim verification under the LMEB "
            "converted retrieval data and scoring setup."
        ),
        eval_langs={"LMEB_SciFact": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/allenai/scifact",
        license="cc-by-nc-3.0",
        domains=["Academic", "Written"],
        task_subtypes=["Claim verification"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class LooGLE(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="LooGLE",
        dataset={
            "path": "mteb/LooGLE",
            "revision": "df52bb043564336892a1eb9127559130fcb0e62c",
        },
        description=(
            "LMEB semantic retrieval task based on LooGLE. Queries ask long- and "
            "short-dependency questions over long documents; the corpus contains "
            "document passages, and the goal is to retrieve the evidence passages "
            "needed to answer each query."
        ),
        eval_langs={subset: ["eng-Latn"] for subset in ["LongDepQA", "ShortDepQA"]},
        reference="https://huggingface.co/datasets/bigai-nlco/LooGLE",
        license="cc-by-sa-4.0",
        domains=["Web", "Written"],
        task_subtypes=["Reading Comprehension"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class LMEBMLDR(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="LMEBMLDR",
        dataset={
            "path": "mteb/LMEBMLDR",
            "revision": "ebc723b2474ef92d098a47e8f131aa7d3542b7b0",
        },
        description=(
            "LMEB semantic retrieval task based on MLDR. Queries are English "
            "long-document question-answering requests; the corpus contains long "
            "document passages, and the goal is to retrieve passages that answer "
            "the query."
        ),
        eval_langs={"MLDR": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/illuin-conteb/mldr-conteb-eval",
        license="mit",
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class ESGReports(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="ESGReports",
        dataset={
            "path": "mteb/ESGReports",
            "revision": "6757300d912abd580dda453bced37cb7f5b44c78",
        },
        description=(
            "LMEB semantic retrieval task based on ESG-Reports. Queries ask "
            "environmental, social, and governance questions; the corpus contains "
            "passages from long-form ESG reports, and the goal is to retrieve "
            "the relevant report evidence."
        ),
        eval_langs={"ESG-Reports": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/illuin-conteb/esg-reports",
        license="not specified",
        domains=["Financial", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class CovidQA(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="CovidQA",
        dataset={
            "path": "mteb/CovidQA",
            "revision": "bf110dabd08865a8aff2d40d03696984e76085e0",
        },
        description=(
            "LMEB semantic retrieval task based on COVID-QA. Queries are COVID-19 "
            "biomedical questions; the corpus contains scientific and medical "
            "evidence passages, and the goal is to retrieve answer-supporting evidence."
        ),
        eval_langs={"Covid-QA": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/illuin-conteb/covid-qa",
        license="apache-2.0",
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class PeerQA(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="PeerQA",
        dataset={
            "path": "mteb/PeerQA",
            "revision": "6b2010c691a435a7a920088d32e55c898afa97f5",
        },
        description=(
            "LMEB semantic retrieval task based on PeerQA. Queries ask questions "
            "about peer reviews and scholarly content; the corpus contains peer-review "
            "material and scholarly writing, and the goal is to retrieve evidence "
            "that answers each query."
        ),
        eval_langs={"PeerQA": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/UKPLab/PeerQA",
        license="cc-by-nc-sa-4.0",
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class NovelQA(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="NovelQA",
        dataset={
            "path": "mteb/NovelQA",
            "revision": "8e4b220f482d1cdceb2f7589900c3ba6cb3cca45",
        },
        description=(
            "LMEB semantic retrieval task based on NovelQA. Queries ask about "
            "characters, meaning, plot, relations, settings, spans, and time in "
            "long-form fiction; the corpus contains novel passages, and the goal "
            "is to retrieve supporting passages."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
                "Character",
                "Meaning",
                "Plot",
                "Relation",
                "Setting",
                "Span",
                "Times",
            ]
        },
        reference="https://huggingface.co/datasets/NovelQA/NovelQA",
        license="not specified",
        domains=["Fiction", "Written"],
        task_subtypes=["Reading Comprehension"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class QASPER(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="QASPER",
        dataset={
            "path": "mteb/QASPER",
            "revision": "898f2eced48890f19b6a88f1b616e9ca6c5dda1c",
        },
        description=(
            "LMEB semantic retrieval task based on QASPER. Queries are "
            "information-seeking questions about research papers; the corpus "
            "contains paper passages, and the goal is to retrieve evidence "
            "passages that support the answer."
        ),
        eval_langs={"QASPER": ["eng-Latn"]},
        reference="https://huggingface.co/datasets/allenai/qasper",
        license="cc-by-nc-4.0",
        domains=["Academic", "Written"],
        task_subtypes=["Reading Comprehension"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class ConvoMem(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="ConvoMem",
        dataset={
            "path": "mteb/ConvoMem",
            "revision": "ae79aabf8da23508a1ba2e982c2552ea14b14a1c",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on ConvoMem. Queries ask "
            "for conversation memories about facts, preferences, abstention, "
            "implicit connections, and changing user state; the corpus contains "
            "dialogue evidence, and the goal is to retrieve the relevant memory snippets."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
                "abstention_evidence",
                "assistant_facts_evidence",
                "changing_evidence",
                "implicit_connection_evidence",
                "preference_evidence",
                "user_evidence",
            ]
        },
        reference="https://huggingface.co/datasets/Salesforce/ConvoMem",
        license="cc-by-nc-4.0",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class MemBench(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="MemBench",
        dataset={
            "path": "mteb/MemBench",
            "revision": "1dd519e4d91573e2818d850eb4405fb290663ac2",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on MemBench. Queries probe "
            "preference, emotion, recommendation, update, aggregation, comparison, "
            "single-hop, and multi-hop memory cues; the corpus contains conversational "
            "memory records, and the goal is to retrieve the relevant memories."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
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
        },
        reference="https://github.com/import-myself/Membench/tree/main/MemData",
        license="mit",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class TMD(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="TMD",
        dataset={
            "path": "mteb/TMD",
            "revision": "7d7b04a28215ca574fcf20e2c42af233864c8e67",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on the Temporal Memory "
            "Dataset. Queries ask date, session, and relative-time questions; "
            "the corpus contains time-anchored dialogue memories, and the goal "
            "is to retrieve the temporal evidence needed for each query."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
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
        },
        reference="https://github.com/Zyphra/TemporalMemoryDataset",
        license="not specified",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class REALTALK(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="REALTALK",
        dataset={
            "path": "mteb/REALTALK",
            "revision": "3ae98b689841a900700ac3594f8e63b8108ae4fe",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on REALTALK. Queries ask "
            "commonsense, multi-hop, and temporal questions over realistic "
            "conversations; the corpus contains conversation evidence, and the "
            "goal is to retrieve supporting turns or memories."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in ["commonsense", "multi_hop", "temporal_reasoning"]
        },
        reference="https://github.com/danny911kr/REALTALK/tree/main/data",
        license="not specified",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class LongMemEval(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="LongMemEval",
        dataset={
            "path": "mteb/LongMemEval",
            "revision": "9dc1a8fdcf9b5676f87c2cdccac021988f6ff5af",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on LongMemEval. Queries "
            "cover single-session, multi-session, preference, knowledge-update, "
            "and temporal reasoning needs; the corpus contains long-term conversation "
            "memory records, and the goal is to retrieve the relevant evidence."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
                "knowledge_update",
                "multi_session",
                "single_session_assistant",
                "single_session_preference",
                "single_session_user",
                "temporal_reasoning",
            ]
        },
        reference="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        license="mit",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class LoCoMo(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="LoCoMo",
        dataset={
            "path": "mteb/LoCoMo",
            "revision": "02e2c3dea15d9fdfd1cd7a0f65f5f8ae2ed4c1ac",
        },
        description=(
            "LMEB dialogue-memory retrieval task based on LoCoMo. Queries are "
            "single-hop, multi-hop, temporal, open-domain, and adversarial "
            "questions over long conversations; the corpus contains conversation "
            "memories, and the goal is to retrieve supporting memories."
        ),
        eval_langs={
            subset: ["eng-Latn"]
            for subset in [
                "single_hop",
                "multi_hop",
                "temporal_reasoning",
                "open_domain",
                "adversarial",
            ]
        },
        reference="https://github.com/snap-research/locomo/tree/main/data",
        license="cc-by-nc-4.0",
        domains=_DIALOGUE_DOMAINS,
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }


class KnowMeBench(AbsTaskRetrieval):
    k_values = _LMEB_K_VALUES

    metadata = TaskMetadata(
        name="KnowMeBench",
        description=(
            "LMEB episodic-memory retrieval task based on KnowMeBench. Queries "
            "require character knowledge, temporal reasoning, mind-body interaction, "
            "information extraction, and adversarial abstention over fictional "
            "narratives; the corpus contains scenario narrative memories, and "
            "the goal is to retrieve the relevant narrative evidence."
        ),
        dataset={
            "path": "mteb/KnowMeBench",
            "revision": "3b817ee68361f2b005bc065910e55f63a8f9ab77",
        },
        eval_langs={
            f"{scenario}__{query_type}": ["eng-Latn"]
            for scenario in _KNOWMEBENCH_SCENARIOS
            for query_type in _KNOWMEBENCH_QUERY_TYPES
        },
        reference="https://github.com/QuantaAlpha/KnowMeBench/tree/main/KnowmeBench",
        license="apache-2.0",
        domains=["Fiction", "Written"],
        task_subtypes=["Question answering"],
        **_common_metadata,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        capped_recall = recall_cap(qrels, results, self.k_values)

        return {
            metric_name: _average_optional(values)
            for metric_name, values in capped_recall.items()
        }
