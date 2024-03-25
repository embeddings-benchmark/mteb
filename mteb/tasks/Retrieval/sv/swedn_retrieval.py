import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class SwednRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SwednRetrieval",
        hf_hub_name="sbx/superlim-2",
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["sv"],
        main_score="ndcg_at_10",
        revision="ef1661775d746e0844b299164773db733bdc0bf6",
        date=("2000-01-01", "2020-12-31"),
        form=["written"],
        domains=["News", "Non-fiction"],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Article retrieval"],
        text_creation="found",
        bibtex_citation="""@inproceedings{monsen2021method,
    title={A method for building non-english corpora for abstractive text summarization},
    author={Monsen, Julius and J{\"o}nsson, Arne},
    booktitle={Proceedings of CLARIN Annual Conference},
    year={2021}
}""",
        n_samples={"test": 2048},
        avg_character_length={"test": 1946.35},
    )

    def load_data(self, **kwargs: dict):  # noqa: ARG002
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset: datasets.DatasetDict = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "swedn",  # chose the relevant subset
            revision=self.metadata_dict.get("revision"),
        )  # type: ignore

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """
        and transform to a retrieval datset, which have the following attributes

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
