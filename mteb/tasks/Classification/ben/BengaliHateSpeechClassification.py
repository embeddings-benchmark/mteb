from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BengaliHateSpeechClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BengaliHateSpeechClassification",
        description="The Bengali Hate Speech Dataset is a Bengali-language dataset of news articles collected from various Bengali media sources and categorized based on the type of hate in the text.",
        reference="https://huggingface.co/datasets/bn_hate_speech",
        dataset={
            "path": "bn_hate_speech",
            "revision": "99612296bc093f0720cac7d7cbfcb67eecf1ca2f",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["ben-Beng"],
        main_score="f1",
        date=("2019-12-01", "2020-04-09"),
        form=["written"],
        dialect=[],
        domains=["News"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        text_creation="found",
        bibtex_citation="""@inproceedings{karim2020BengaliNLP,
    title={Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
    author={Karim, Md. Rezaul and Chakravarti, Bharathi Raja and P. McCrae, John and Cochez, Michael},
    booktitle={7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
    publisher={IEEE},
    year={2020}
}
""",
        n_samples={"train": 3418},
        avg_character_length={"train": 103.42},
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train"].select(range(2048))
