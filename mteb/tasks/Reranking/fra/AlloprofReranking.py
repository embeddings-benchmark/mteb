from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class AlloprofReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AlloprofReranking",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-alloprof-s2p",
            "revision": "e40c8a63ce02da43200eccb5b0846fcaa888f562", # TODO : change
        },
        type="Reranking",
        category="s2s",
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
        n_samples={"test": 2316, "train": 9264},
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