from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GreekCivicsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreekCivicsQA",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "ilsp/greek_civics_qa",
            "revision": "a04523a3c83153be07a8945bb1fb351cbbcef90b",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["default"],
        eval_langs=["ell-Grek"],
        main_score="ndcg_at_10",
        date=("2023-01-01","2024-04-01"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"default": 407},
        avg_character_length={"default": 2226.85},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        data_raw = datasets.load_dataset(**self.metadata_dict["dataset"])
        eval_split = self.metadata_dict["eval_splits"][0]
        self.queries = {
            eval_split: {f"q{q['id']}": q["question"] for q in data_raw[eval_split]}
        }
        self.corpus = {
            eval_split: {
                f"d{d['id']}": {"text": d["answer"], "title": d['answer']} for d in data_raw[eval_split]
            }
        }

        self.relevant_docs = {eval_split: {f"q{i+1}": {f"d{i+1}": 1} for i in range(data_raw[eval_split].shape[0])}}

        self.data_loaded = True
