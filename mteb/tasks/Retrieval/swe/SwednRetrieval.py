import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class SwednRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SwednRetrieval",
        dataset={
            "path": "sbx/superlim-2",
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
            "name": "swedn",
            "trust_remote_code": True,
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2020-12-31"),
        domains=["News", "Non-fiction", "Written"],
        license="CC BY-SA 4.0",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Article retrieval"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{monsen2021method,
    title={A method for building non-english corpora for abstractive text summarization},
    author={Monsen, Julius and J{\"o}nsson, Arne},
    booktitle={Proceedings of CLARIN Annual Conference},
    year={2021}
}""",
        descriptive_stats={
            "n_samples": {"test": 2048},
            "avg_character_length": {
                "test": {
                    "average_document_length": 2896.519550342131,
                    "average_query_length": 45.876953125,
                    "num_documents": 2046,
                    "num_queries": 1024,
                    "average_relevant_docs_per_query": 2.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            ds = ds.select(
                range(1024)
            )  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            headline = ds["headline"]
            summary = ds["summary"]
            article = ds["article"]

            n = 0
            for headl, summ, art in zip(headline, summary, article):
                self.queries[split][str(n)] = headl
                q_n = n
                n += 1
                if summ not in text2id:
                    text2id[summ] = n
                    self.corpus[split][str(n)] = {"title": "", "text": summ}
                    n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[art]): 1,
                    str(text2id[summ]): 1,
                }  # only two correct matches
