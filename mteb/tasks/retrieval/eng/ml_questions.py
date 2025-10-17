import csv
from pathlib import Path

from huggingface_hub import snapshot_download

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MLQuestionsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MLQuestions",
        dataset={
            "path": "McGill-NLP/mlquestions",
            "revision": "83b690cb666c5a8869e7f213a877bbd24a642d7c",
        },
        reference="https://github.com/McGill-NLP/MLQuestions",
        description=(
            "MLQuestions is a domain adaptation dataset for the machine learning domain"
            + "It consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)"
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2021-01-01",
            "2021-03-31",
        ),  # The period here is for both wiki articles and queries
        domains=["Encyclopaedic", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kulshreshtha-etal-2021-back,
  address = {Online and Punta Cana, Dominican Republic},
  author = {Kulshreshtha, Devang  and
Belfer, Robert  and
Serban, Iulian Vlad  and
Reddy, Siva},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  month = nov,
  pages = {7064--7078},
  publisher = {Association for Computational Linguistics},
  title = {Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval},
  url = {https://aclanthology.org/2021.emnlp-main.566},
  year = {2021},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset.get("revision", None)
        download_dir = snapshot_download(
            repo_id=dataset_path, repo_type="dataset", revision=revision
        )
        for split in self.metadata.eval_splits:
            corpus, queries, qrels = self._load_data_for_split(download_dir, split)
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True

    def _load_data_for_split(self, download_dir, split):
        queries, corpus, qrels = {}, {}, {}

        download_dir = Path(download_dir)
        dataset_path = download_dir / f"{split}.csv"
        with dataset_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                query_id = f"Q{str(i)}"
                doc_id = row["indexes"]
                query = row["target_text"]
                queries[query_id] = query
                qrels[query_id] = {f"C{doc_id}": 1}

        # Same corpus for all splits
        corpus_path = download_dir / "test_passages.csv"
        with corpus_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                doc_id = f"C{str(i)}"
                corpus[doc_id] = {
                    "title": "",
                    "text": row["input_text"],
                }

        return corpus, queries, qrels
