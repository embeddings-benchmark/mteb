import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class SNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SNLRetrieval",
        dataset={
            "path": "navjordj/SNL_summarization",
            "revision": "3d3d27aa7af8941408cefc3991ada5d12a4273d1",
        },
        description="Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.",
        reference="https://huggingface.co/datasets/navjordj/SNL_summarization",
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        form=["written"],
        domains=["Encyclopaedic", "Non-fiction"],
        license=None,
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@mastersthesis{navjord2023beyond,
    title={Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
    author={Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
    year={2023},
    school={Norwegian University of Life Sciences, {\AA}s}
}""",
        n_samples={"test": 2048},
        avg_character_length={"test": 1101.30},
        task_subtypes=["Article retrieval"],
    )

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

            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            headline = ds["headline"]
            article = ds["article"]

            n = 0
            for headl, art in zip(headline, article):
                self.queries[split][str(n)] = headl
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1
                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[art]): 1
                }  # only one correct matches
