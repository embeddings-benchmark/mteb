import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BBSARDNLRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="bBSARDNLRetrieval",
        description="Building on the Belgian Statutory Article Retrieval Dataset (BSARD) in French, we introduce the "
        "bilingual version of this dataset, bBSARD. The dataset contains parallel Belgian statutory "
        "articles in both French and Dutch, along with legal questions from BSARD and their Dutch "
        "translation.",
        reference="https://aclanthology.org/2025.regnlp-1.3.pdf",
        dataset={
            "path": "clips/bBSARD",
            "revision": "3f2554a64f15cabdccd72844008200e9a9279100",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lotfi2025bilingual,
  author = {Lotfi, Ehsan and Banar, Nikolay and Yuzbashyan, Nerses and Daelemans, Walter},
  journal = {COLING 2025},
  pages = {10},
  title = {Bilingual BSARD: Extending Statutory Article Retrieval to Dutch},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            split="nl",
            **self.metadata.dataset,
            # **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name=self.metadata.eval_splits[0],
            split="nl",
            **self.metadata.dataset,
            # **self.metadata_dict["dataset"],
        )

        self.queries = {
            self.metadata.eval_splits[0]: {
                str(q["id"]): q["question"].strip() for q in queries_raw
            }
        }

        self.corpus = {
            self.metadata.eval_splits[0]: {
                str(d["id"]): {"text": d["article"]} for d in corpus_raw
            }
        }

        self.relevant_docs = {"test": {str(q["id"]): {} for q in queries_raw}}
        for q in queries_raw:
            for doc_id in q["article_ids"].split(","):
                self.relevant_docs[self.metadata.eval_splits[0]][str(q["id"])][
                    doc_id
                ] = 1

        self.data_loaded = True
