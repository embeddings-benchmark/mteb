from logging import getLogger

import datasets

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = getLogger(__name__)

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


def load_data(
    path: str,
    langs: list,
    eval_splits: list,
    revision: str | None = None,
):
    corpus = {lang: {EVAL_SPLIT: {}} for lang in langs}
    queries = {lang: {EVAL_SPLIT: {}} for lang in langs}
    relevant_docs = {lang: {EVAL_SPLIT: {}} for lang in langs}
    instructions = {lang: {EVAL_SPLIT: {}} for lang in langs}
    top_ranked = {lang: {EVAL_SPLIT: {}} for lang in langs}
    qrel_diffs = {lang: {EVAL_SPLIT: {}} for lang in langs}

    for lang in langs:
        if "-" in lang:
            loading_lang = lang.split("-")[1]  # don't care about the eng part
        else:
            loading_lang = lang
        logger.info(f"Loading data for {lang} from {loading_lang}")

        # Load corpus data
        corpus_data = datasets.load_dataset(
            path,
            f"corpus-{loading_lang}",
            revision=revision,
        )
        corpus[lang][EVAL_SPLIT] = {
            row["_id"]: {"title": row["title"], "text": row["text"]}
            for row in corpus_data["corpus"]
        }

        # Load queries data
        queries_data = datasets.load_dataset(
            path,
            f"queries-{loading_lang}",
            revision=revision,
        )
        queries[lang][EVAL_SPLIT] = {
            row["_id"]: row["text"] for row in queries_data["queries"]
        }

        # Load instructions data
        instructions_data = datasets.load_dataset(
            path,
            f"instruction-{loading_lang}",
            revision=revision,
        )
        instructions[lang][EVAL_SPLIT] = {
            row["query-id"]: row["instruction"]
            for row in instructions_data["instruction"]
        }

        # Load qrels_og data
        qrels_og_data = datasets.load_dataset(
            path,
            f"default-{loading_lang}",
            revision=revision,
        )
        for row in qrels_og_data[EVAL_SPLIT]:
            if row["query-id"] not in relevant_docs[lang][EVAL_SPLIT]:
                relevant_docs[lang][EVAL_SPLIT][row["query-id"]] = {
                    row["corpus-id"]: int(row["score"])
                }
            else:
                relevant_docs[lang][EVAL_SPLIT][row["query-id"]][row["corpus-id"]] = (
                    int(row["score"])
                )

        # Load top_ranked data
        top_ranked_data = datasets.load_dataset(
            path,
            f"top_ranked-{loading_lang}",
            revision=revision,
        )
        for row in top_ranked_data["top_ranked"]:
            top_ranked[lang][EVAL_SPLIT][row["query-id"]] = row["corpus-ids"]

        qrel_diff = datasets.load_dataset(
            path,
            f"qrel_diff-{loading_lang}",
            split="qrel_diff",
            revision=revision,
        )

        qrel_diffs[lang][EVAL_SPLIT] = {
            item["query-id"]: item["corpus-ids"] for item in qrel_diff
        }

    return corpus, queries, instructions, relevant_docs, top_ranked, qrel_diffs


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
            "path": "jhu-clsp/mFollowIR-cross-lingual-parquet-mteb",
            "revision": "6b01566619233a0c35d135123510b6b02c258ff5",
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
@article{weller2024mfollowir,
  author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Sam and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
  journal = {arXiv preprint TODO},
  title = {{mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieval}},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        (
            self.corpus,
            self.queries,
            self.instructions,
            self.relevant_docs,
            self.top_ranked,
            self.qrels_diff,
        ) = load_data(
            path=self.metadata.dataset["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

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
            "path": "jhu-clsp/mFollowIR-parquet-mteb",
            "revision": "09eecbe45c54b4a6dfb8e68e345cae77337768e2",
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
@article{weller2024mfollowir,
  author = {Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Sam and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
  journal = {arXiv preprint TODO},
  title = {{mFollowIR: a Multilingual Benchmark for Instruction Following in Retrieval}},
  year = {2024},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        (
            self.corpus,
            self.queries,
            self.instructions,
            self.relevant_docs,
            self.top_ranked,
            self.qrels_diff,
        ) = load_data(
            path=self.metadata.dataset["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

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
