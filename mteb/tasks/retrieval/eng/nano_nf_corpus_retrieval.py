from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoNFCorpusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNFCorpusRetrieval",
        description="NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        dataset={
            "path": "zeta-alpha-ai/NanoNFCorpus",
            "revision": "dd542a7efb9ad2136b9e00768b60fca9038f8156",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2016-01-01", "2016-12-31"],
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  city = {Padova},
  country = {Italy},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
  year = {2016},
}
""",
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
        adapted_from=["NFCorpus"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "corpus",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "queries",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoNFCorpus",
            "qrels",
            revision="dd542a7efb9ad2136b9e00768b60fca9038f8156",
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
