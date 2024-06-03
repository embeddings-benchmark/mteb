from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FaithDialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FaithDial",
        dataset={
            "path": "McGill-NLP/FaithDial",
            "revision": "7a414e80725eac766f2602676dc8b39f80b061e4",
        },
        reference="https://mcgill-nlp.github.io/FaithDial",
        description=(
            "FaithDial is a faithful knowledge-grounded dialogue benchmark."
            "It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW)."
            "It consists of conversation histories along with manually labelled relevant passage."
            "For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field,"
            "as the gold passage associated with these instances is non-ambiguous."
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-03-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @article{dziri2022faithdial,
            title = "{FaithDial: A Faithful Benchmark for Information-Seeking Dialogue}",
            author = {Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo M and Reddy, Siva},
            journal = {Transactions of the Association for Computational Linguistics},
            volume = {10},
            pages = {1473--1490},
            year = {2022},
            month = {12},
            publisher = {MIT Press},
            doi={10.1162/tacl_a_00529}
            }
        """,
        n_samples=None,  # TODO: Add n_samples
        avg_character_length=None,  # TODO: Add avg_character_length
    )

    # TODO: Will be removed if curated and added to mteb HF
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            corpus, queries, qrels = self._load_data_for_split(dataset_path, split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, dataset_path, split):
        revision = self.metadata_dict["dataset"].get("revision", None)
        ds = load_dataset(
            dataset_path,
            split=split,
            revision=revision,
        )
        queries, corpus, qrels = {}, {}, {}
        for i, sample in enumerate(ds):
            # document is added to corpus for all samples
            doc_id = "doc:" + str(i)
            corpus[doc_id] = {
                "title": "",  # title is not available
                "text": sample["knowledge"],
            }
            if "Edification" in sample["VRM"]:
                query_id = "query:" + str(i)
                query = sample["history"]
                queries[query_id] = query
                qrels[query_id] = {doc_id: 1}

        return corpus, queries, qrels
