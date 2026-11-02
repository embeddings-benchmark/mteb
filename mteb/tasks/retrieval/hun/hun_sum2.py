from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HunSum2AbstractiveRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HunSum2AbstractiveRetrieval",
        dataset={
            "path": "SZTAKI-HLT/HunSum-2-abstractive",
            "revision": "24e1445c8180d937f0a16f8ae8a62e77cc952e56",
        },
        description=(
            "HunSum-2-abstractive is a Hungarian dataset containing news articles along with lead, titles and metadata."
        ),
        reference="https://aclanthology.org/2024.lrec-main.662/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hun-Latn"],
        main_score="ndcg_at_1",
        date=(
            "1848-12-15",
            "2024-03-19",
        ),
        domains=["News", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{barta2024news,
  address = {Torino, Italia},
  archiveprefix = {arXiv},
  author = {Barta, Botond and Lakatos, Dorina and Nagy, Attila and Nyist, Mil{\'a}n Konor and {\'A}cs, Judit},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation ({LREC}-{COLING} 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  eprint = {2404.03555},
  month = {may},
  pages = {7503--7509},
  publisher = {ELRA and ICCL},
  title = {From News to Summaries: Building a {H}ungarian Corpus for Extractive and Abstractive Summarization},
  url = {https://aclanthology.org/2024.lrec-main.662/},
  year = {2024},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        ds = load_dataset(**self.metadata.dataset, split=self.metadata.eval_splits)
        ds = dict(zip(self.metadata.eval_splits, ds, strict=True))
        for split_name, split in ds.items():
            self.corpus[split_name] = {}
            self.queries[split_name] = {}
            self.relevant_docs[split_name] = {}
            for record in split:
                self.corpus[split_name]["d" + record["uuid"]] = {
                    "title": record["title"],
                    "text": record["article"],
                }
                self.queries[split_name]["q" + record["uuid"]] = record["lead"]
                self.relevant_docs[split_name]["q" + record["uuid"]] = {
                    "d" + record["uuid"]: 1
                }
        self.data_loaded = True
