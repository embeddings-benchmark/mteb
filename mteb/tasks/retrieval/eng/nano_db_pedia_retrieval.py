from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoDBPediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPediaRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.",
        reference="https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia",
        dataset={
            "path": "zeta-alpha-ai/NanoDBPedia",
            "revision": "438f1c25129f05db6238699b5afdc9c6b58d2096",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2015-01-01", "2015-12-31"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lehmann2015dbpedia,
  author = {Lehmann, Jens and et al.},
  journal = {Semantic Web},
  title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},
  year = {2015},
}
""",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        adapted_from=["DBPedia"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "corpus",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "queries",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoDBPedia",
            "qrels",
            revision="438f1c25129f05db6238699b5afdc9c6b58d2096",
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
