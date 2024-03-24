from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BSARDRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BSARDRetrieval",
        description="The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.",
        reference="https://huggingface.co/datasets/maastrichtlawtech/bsard",
        hf_hub_name="mteb/bsard",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fr"],
        main_score="ndcg_at_100",
        revision="5effa1b9b5fa3b0f9e12523e6e43e5f86a6e6d59",
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
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "corpus",
            split="corpus",
            revision=self.metadata_dict.get("revision", None),
        )
        queries_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            "questions",
            split=self.metadata.eval_splits[0],
            revision=self.metadata_dict.get("revision", None),
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
