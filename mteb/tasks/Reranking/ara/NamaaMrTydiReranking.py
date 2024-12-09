from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking

import datasets

class NamaaMrTydiReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="NamaaMrTydiReranking",
        description="MrTydi reranking dataset for arabic reranking evaluation",
        reference="https://huggingface.co/NAMAA-Space",
        dataset={
            "path": "NAMAA-Space/mteb-eval-mrtydi",
            "revision": "bb3638ffe3b2be76fe2e5a4581123923afee5cda",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ara-Arab"],
        main_score="map",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        prompt="",
        bibtex_citation="",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            name="default",
            **self.metadata_dict["dataset"],
            split=self.metadata.eval_splits[0],
        )

        self.dataset = self.dataset.map(
            lambda x: {
                "query" : x["query"],
                "positive": [x["positive"]],
                "negative": x["negative"],
            }
        )
        self.dataset = datasets.DatasetDict({"test": self.dataset})
        self.dataset_transform()

        self.data_loaded = True