import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SNLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SNLRetrieval",
        dataset={
            "path": "adrlau/navjordj-SNL_summarization_copy",  # TODO: replace with mteb/SNLRetrieval after #2820 is resolved.
            "revision": "22c474c88fb4678052f9099bb917ad8f9e155f9f",
        },
        description="Webscrabed articles and ingresses from the Norwegian lexicon 'Det Store Norske Leksikon'.",
        reference="https://huggingface.co/datasets/mteb/SNLRetrieval",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-nc-4.0",  # version assumed (not specified beforehand)
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@mastersthesis{navjord2023beyond,
  author = {Navjord, J{\o}rgen Johnsen and Korsvik, Jon-Mikkel Ryen},
  school = {Norwegian University of Life Sciences, {\AA}s},
  title = {Beyond extractive: advancing abstractive automatic text summarization in Norwegian with transformers},
  year = {2023},
}
""",
        prompt={"query": "Given a lexicon headline in Norwegian, retrieve its article"},
        task_subtypes=["Article retrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        self.dataset = {}
        text2id = {}

        for split in hf_dataset:
            ds: datasets.Dataset = hf_dataset[split]
            ds = ds.shuffle(seed=42)

            queries_dict = {}
            corpus_dict = {}
            relevant_docs = {}

            headline = ds["headline"]
            article = ds["article"]

            n = 0
            for headl, art in zip(headline, article):
                queries_dict[str(n)] = headl
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    corpus_dict[str(n)] = {"title": "", "text": art}
                    n += 1
                relevant_docs[str(q_n)] = {
                    str(text2id[art]): 1
                }  # only one correct matches

            corpus_dataset = Dataset.from_list(
                [
                    {"id": k, "text": v["text"], "title": v["title"]}
                    for k, v in corpus_dict.items()
                ]
            )
            queries_dataset = Dataset.from_list(
                [{"id": k, "text": v} for k, v in queries_dict.items()]
            )

            self.dataset.setdefault("default", {})[split] = {
                "corpus": corpus_dataset,
                "queries": queries_dataset,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
