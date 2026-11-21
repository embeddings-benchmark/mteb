from __future__ import annotations

import datasets

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}

_LANGUAGES_CLIR = {
    "eng.fas": ["eng-Latn", "fas-Arab"],
    "eng.rus": ["eng-Latn", "rus-Cyrl"],
    "eng.zho": ["eng-Latn", "zho-Hans"],
}


def _build_lang_pair(langs: list[str]) -> str:
    """Builds a language pair separated by a dash.
    e.g., ['eng-Latn', 'deu-Latn'] -> 'eng-deu'.
    """
    return langs[0].split("-")[0] + "-" + langs[1].split("-")[0]


def extend_lang_pairs() -> dict[str, list[str]]:
    eval_langs = {}
    for langs in _LANGUAGES_CLIR.values():
        lang_pair = _build_lang_pair(langs)
        eval_langs[lang_pair] = langs
    return eval_langs


_CLIR_LANGS = extend_lang_pairs()

EVAL_SPLIT = "test"


def load_qrel_diff(metadata: TaskMetadata, hf_subset: str) -> dict[str, list[str]]:
    hf_subset = hf_subset.replace("eng-", "")
    qrel_diff_ds = datasets.load_dataset(
        metadata.dataset["path"],
        f"qrel_diff-{hf_subset}",
        split="qrel_diff",
        revision=metadata.dataset["revision"],
    )
    return {item["query-id"]: item["corpus-ids"] for item in qrel_diff_ds}


class mFollowIRCrossLingual(AbsTaskRetrieval):  # noqa: N801
    metadata = TaskMetadata(
        name="mFollowIRCrossLingual",
        description="This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages with English queries/instructions.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/mFollowIRCrossLingual",
            "revision": "2882dd6076298fc65a3515f10f8924e806ffa4f8",
        },
        type="InstructionReranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[EVAL_SPLIT],
        eval_langs=_CLIR_LANGS,
        main_score="p-MRR",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{weller2024mfollowir,
  author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Samuel and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
  booktitle = {Advances in Information Retrieval},
  doi = {10.1007/978-3-031-88711-6_19},
  isbn = {9783031887116},
  issn = {1611-3349},
  pages = {295–310},
  publisher = {Springer Nature Switzerland},
  title = {mFollowIR: A Multilingual Benchmark for Instruction Following in Retrieval},
  url = {https://doi.org/10.1007/978-3-031-88711-6_19},
  year = {2025},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return evaluate_p_mrr_change(
            qrels,
            results,
            load_qrel_diff(self.metadata, hf_subset),
            self.k_values,
        )


class mFollowIR(AbsTaskRetrieval):  # noqa: N801
    metadata = TaskMetadata(
        name="mFollowIR",
        description="This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/mFollowIR",
            "revision": "86d5e6d63e148af1cf9c85139c4796cae0a21feb",
        },
        type="InstructionReranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="p-MRR",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{weller2024mfollowir,
  author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Samuel and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
  booktitle = {Advances in Information Retrieval},
  doi = {10.1007/978-3-031-88711-6_19},
  isbn = {9783031887116},
  issn = {1611-3349},
  pages = {295–310},
  publisher = {Springer Nature Switzerland},
  title = {mFollowIR: A Multilingual Benchmark for Instruction Following in Retrieval},
  url = {https://doi.org/10.1007/978-3-031-88711-6_19},
  year = {2025},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return evaluate_p_mrr_change(
            qrels,
            results,
            load_qrel_diff(self.metadata, hf_subset),
            self.k_values,
        )
