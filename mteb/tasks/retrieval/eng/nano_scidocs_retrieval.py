from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoSCIDOCSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSCIDOCSRetrieval",
        description="NanoFiQA2018 is a smaller subset of "
        + "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
        + " prediction, to document classification and recommendation.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "zeta-alpha-ai/NanoSCIDOCS",
            "revision": "484eb90549fc3f0b9c42b3551e80ceb999515537",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
        adapted_from=["SCIDOCS"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "corpus",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "queries",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoSCIDOCS",
            "qrels",
            revision="484eb90549fc3f0b9c42b3551e80ceb999515537",
        )

        self.dataset = {}
        for split in corpus_hf:
            corpus_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in corpus_hf[split]]
            )
            queries_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in queries_hf[split]]
            )
            relevant_docs: dict = defaultdict(dict)
            for query_id, corpus_id in zip(
                qrels_hf[split]["query-id"],
                qrels_hf[split]["corpus-id"],
            ):
                relevant_docs[query_id][corpus_id] = 1
            if "default" not in self.dataset:
                self.dataset["default"] = {}
            self.dataset["default"][split] = {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": dict(relevant_docs),
                "top_ranked": None,
            }

        self.data_loaded = True
