from datasets import load_dataset

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RuFaithDialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuFaithDialRetrieval",
        dataset={
            "path": "DeepPavlov/FaithDial-ru",
            "revision": "ed49d9732196e96d5291e11cfa416083b8ff699e",
        },
        reference="https://mcgill-nlp.github.io/FaithDial",
        description=(
            "FaithDial is a faithful knowledge-grounded dialogue benchmark."
            + "It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). "
            + "It consists of conversation histories along with manually labelled relevant passage. "
            + "For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, "
            + "as the gold passage associated with these instances is non-ambiguous."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-03-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""
@article{dziri2022faithdial,
  author = {Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo M and Reddy, Siva},
  doi = {10.1162/tacl_a_00529},
  journal = {Transactions of the Association for Computational Linguistics},
  month = {12},
  pages = {1473--1490},
  publisher = {MIT Press},
  title = {{FaithDial: A Faithful Benchmark for Information-Seeking Dialogue}},
  volume = {10},
  year = {2022},
}
""",
    )

    # TODO: Will be removed if curated and added to mteb HF
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in kwargs.get("eval_splits", self.metadata.eval_splits):
            corpus, queries, qrels = self._load_data_for_split(split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, split):
        ds = load_dataset(split=split, **self.metadata.dataset)
        queries, corpus, qrels = {}, {}, {}
        for i, sample in enumerate(ds):
            # document is added to corpus for all samples
            doc_id = "doc:" + str(i)
            corpus[doc_id] = {
                "title": "",  # title is not available
                "text": sample["knowledge_ru"],
            }
            if "Edification" in sample["VRM"]:
                query_id = "query:" + str(i)
                query = sample["history"]
                queries[query_id] = query
                qrels[query_id] = {doc_id: 1}

        return corpus, queries, qrels
