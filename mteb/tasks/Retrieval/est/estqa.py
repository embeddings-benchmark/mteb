from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class EstQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EstQA",
        dataset={
            "path": "kardosdrur/estonian-qa",
            "revision": "766c9bc74b70e6bcdd5a7720a9c7556ac10a8b04",
        },
        description=(
            "EstQA is an Estonian question answering dataset based on Wikipedia."
        ),
        reference="https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link",
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "test"],
        eval_langs=["est-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2002-08-24",
            "2021-05-10",
        ),  # birth of Estonian Wikipedia to publishing the article
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@mastersthesis{mastersthesis,
  author       = {Anu Käver},
  title        = {Extractive Question Answering for Estonian Language},
  school       = {Tallinn University of Technology (TalTech)},
  year         = 2021
}
""",
        n_samples={"train": 482, "test": 121},
        avg_character_length={"train": 772.5331950207469, "test": 770.9834710743802},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        ds = load_dataset(**self.metadata.dataset)
        for split_name, split in ds.items():
            self.corpus[split_name] = {}
            self.queries[split_name] = {}
            self.relevant_docs[split_name] = {}
            for record in split:
                self.corpus[split_name]["d" + record["id"]] = {
                    "title": record["title"],
                    "text": record["context"],
                }
                self.queries[split_name]["q" + record["id"]] = record["question"]
                self.relevant_docs[split_name]["q" + record["id"]] = {
                    "d" + record["id"]: 1
                }
        self.data_loaded = True
