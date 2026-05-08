import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CQADupstackMathematicaNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackMathematica-NL",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research. This a "
        "Dutch-translated version.",
        reference="https://huggingface.co/datasets/clips/beir-nl-cqadupstack",
        dataset={
            "path": "clips/beir-nl-cqadupstack",
            "revision": "1de13232f452f0b4f5525d8c6c41cd6a9ae1d084",
            "split": "mathematica",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2015-12-01", "2015-12-01"),  # best guess: based on publication date
        domains=["Written", "Non-fiction"],
        task_subtypes=["Duplicate Detection"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",  # manually checked a small subset
        bibtex_citation=r"""
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
        adapted_from=["CQADupstackMathematicaRetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_raw = datasets.load_dataset(
            name="corpus",
            **self.metadata.dataset,
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata.dataset,
        )
        qrels_raw = datasets.load_dataset(
            name="test",
            **self.metadata.dataset,
        )

        queries_ds = Dataset.from_list(
            [{"id": q["_id"], "text": q["text"]} for q in queries_raw]
        )
        corpus_ds = Dataset.from_list(
            [
                {"id": d["_id"], "text": d["text"], "title": d["title"]}
                for d in corpus_raw
            ]
        )
        relevant_docs = {}
        for q in qrels_raw:
            qid = q["query-id"]
            if qid not in relevant_docs:
                relevant_docs[qid] = {}
            relevant_docs[qid][q["corpus-id"]] = int(q["score"])

        self.dataset = {
            "default": {
                "test": {
                    "corpus": corpus_ds,
                    "queries": queries_ds,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True
