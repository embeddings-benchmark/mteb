from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SyntecReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SyntecReranking",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "revision": "d99977875bf70d85753cebd7fcb61bef330ceec9",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
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
        n_samples={"test": 100},
        avg_character_length=None,
    )

    def load_data(self):

        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
            split=self.metadata["eval_splits"][0]
        )
        self.documents = datasets.load_dataset(
            name="documents",
            **self.metadata_dict["dataset"],
            split="test"
        )
        # replace documents ids in positive and negative column by their respective texts
        self.dataset = self.dataset.map(lambda x: {
            "positive": [
                self.documents["text"][self.documents["doc_id"].index(docid)]
                for docid in x["positive"]
            ],
            "negative": [
                self.documents["text"][self.documents["doc_id"].index(docid)]
                for docid in x["negative"]
            ]
        })

        self.data_loaded = True
