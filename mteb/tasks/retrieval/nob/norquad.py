import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NorQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NorQuadRetrieval",
        dataset={
            "path": "mteb/norquad_retrieval",
            "revision": "9dcfcdb2aa578dd178330d49bf564248935f7fbe",
        },
        description="Human-created question for Norwegian wikipedia passages.",
        reference="https://aclanthology.org/2023.nodalida-1.17/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2023-12-31"),
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ivanova-etal-2023-norquad,
  address = {T{\'o}rshavn, Faroe Islands},
  author = {Ivanova, Sardana  and
Andreassen, Fredrik  and
Jentoft, Matias  and
Wold, Sondre  and
{\O}vrelid, Lilja},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Alum{\"a}e, Tanel  and
Fishel, Mark},
  month = may,
  pages = {159--168},
  publisher = {University of Tartu Library},
  title = {{N}or{Q}u{AD}: {N}orwegian Question Answering Dataset},
  url = {https://aclanthology.org/2023.nodalida-1.17},
  year = {2023},
}
""",
        prompt={
            "query": "Given a question in Norwegian, retrieve the answer from Wikipedia articles"
        },
    )

    def load_data(self) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.corpus = dict[doc_id, dict[str, str]] #id => dict with document data like title and text
        self.queries = dict[query_id, str] #id => query
        self.relevant_docs = dict[query_id, dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
            ds = ds.shuffle(seed=42)
            max_samples = min(1024, len(ds))
            ds = ds.select(
                range(max_samples)
            )  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            question = ds["question"]
            context = ds["context"]
            answer = [a["text"][0] for a in ds["answers"]]

            n = 0
            for q, cont, ans in zip(question, context, answer):
                self.queries[split][str(n)] = q
                q_n = n
                n += 1
                if cont not in text2id:
                    text2id[cont] = n
                    self.corpus[split][str(n)] = {"title": "", "text": cont}
                    n += 1
                if ans not in text2id:
                    text2id[ans] = n
                    self.corpus[split][str(n)] = {"title": "", "text": ans}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[ans]): 1,
                    str(text2id[cont]): 1,
                }  # only two correct matches
