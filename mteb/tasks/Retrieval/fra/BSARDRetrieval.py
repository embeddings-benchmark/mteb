from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BSARDRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BSARDRetrieval",
        description="The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.",
        reference="https://huggingface.co/datasets/maastrichtlawtech/bsard",
        dataset={
            "path": "maastrichtlawtech/bsard",
            "revision": "5effa1b9b5fa3b0f9e12523e6e43e5f86a6e6d59",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="ndcg_at_100",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name="questions",
            split=self.metadata.eval_splits[0],
            **self.metadata_dict["dataset"],
        )

        self.queries = {
            self.metadata.eval_splits[0]: {
                str(q["id"]): " ".join((q["question"] + q["extra_description"]))
                for q in queries_raw
            }
        }

        self.corpus = {
            self.metadata.eval_splits[0]: {
                str(d["id"]): {"text": d["article"]} for d in corpus_raw
            }
        }

        self.relevant_docs = {self.metadata.eval_splits[0]: {}}
        for q in queries_raw:
            for doc_id in q["article_ids"]:
                self.relevant_docs[self.metadata.eval_splits[0]][str(q["id"])] = {
                    str(doc_id): 1
                }

        self.data_loaded = True
