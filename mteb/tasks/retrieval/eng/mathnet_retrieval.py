from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Features, Value, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

_TIERS = ["easy", "medium", "hard"]


class MathNetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MathNetRetrieval",
        description=(
            "The Math-Aware Retrieval benchmark from MathNet: given an Olympiad problem, retrieve a mathematically "
            "equivalent reformulation of it from a shared corpus of 117,088 documents containing equivalents at all "
            "difficulty tiers, LLM-generated near-miss hard negatives and distractors. Each of the 15,000 queries has "
            "exactly one gold equivalent per tier. The easy, medium and hard subsets increasingly disguise the surface "
            "form of the problem (from light paraphrase to heavy disguise with minimal lexical overlap) while "
            "preserving the underlying mathematics; the query's equivalents at the other tiers act as negatives. "
            "Problems are drawn from national and international olympiads; most text is English, with a small "
            "fraction also containing the original-language statement (e.g. Russian or Chinese)."
        ),
        reference="https://arxiv.org/abs/2604.18584",
        dataset={
            "path": "ShadenA/MathNet-Retrieve",
            "revision": "edab33b5ae2c56c18da0b9a174530de2012a1075",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={tier: ["eng-Latn"] for tier in _TIERS},
        main_score="ndcg_at_10",
        date=("1961-01-01", "2025-12-31"),
        domains=["Academic", "Written"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=r"""
@inproceedings{alshammari2026mathnet,
  author = {Alshammari, Shaden and Wen, Kevin and Zainal, Abrar and Hamilton, Mark and Safaei, Navid and Albarakati, Sultan and Freeman, William T. and Torralba, Antonio},
  booktitle = {International Conference on Learning Representations},
  title = {MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval},
  url = {https://mathnet.mit.edu},
  year = {2026},
}
""",
        prompt={
            "query": "Given a math olympiad problem, retrieve a mathematically equivalent problem"
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        # The Hub dataset stores each tier as a config whose splits (`corpus`,
        # `queries`, `qrels`) have different schemas, so the files are loaded
        # individually. Corpus and queries are identical across tiers.
        def _load(file: str) -> Dataset:
            return load_dataset(
                self.metadata.dataset["path"],
                data_files=file,
                split="train",
                revision=self.metadata.dataset["revision"],
                num_proc=num_proc,
            )

        corpus = _load("easy/corpus.jsonl").rename_column("_id", "id")
        queries = _load("easy/queries.jsonl").rename_column("_id", "id")

        self.dataset = {}
        for tier in self.hf_subsets:
            qrels = (
                _load(f"{tier}/qrels/test.jsonl")
                .select_columns(["query-id", "corpus-id", "score"])
                .cast(
                    Features(
                        {
                            "query-id": Value("string"),
                            "corpus-id": Value("string"),
                            "score": Value("int32"),
                        }
                    )
                )
            )
            relevant_docs: dict[str, dict[str, int]] = {}
            for row in qrels:
                relevant_docs.setdefault(row["query-id"], {})[row["corpus-id"]] = row[
                    "score"
                ]
            self.dataset[tier] = {
                split: RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
                for split in self.eval_splits
            }

        self.dataset_transform(num_proc=num_proc)
        self.data_loaded = True
