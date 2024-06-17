from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class FilipinoHateSpeechClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FilipinoHateSpeechClassification",
        description="Filipino Twitter dataset for sentiment classification.",
        reference="https://pcj.csp.org.ph/index.php/pcj/issue/download/29/PCJ%20V14%20N1%20pp1-14%202019",
        dataset={
            "path": "hate-speech-filipino/hate_speech_filipino",
            "revision": "1994e9bb7f3ec07518e3f0d9e870cb293e234686",
        },
        type="Classification",
        category="s2s",
        date=("2019-08-01", "2019-08-01"),
        eval_splits=["validation", "test"],
        eval_langs=["fil-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @article{Cabasag-2019-hate-speech,
            title={Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
            author={Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
            journal={Philippine Computing Journal},
            volume={XIV},
            number={1},
            month={August},
            year={2019}
        }
        """,
        n_samples={"validation": TEST_SAMPLES, "test": TEST_SAMPLES},
        avg_character_length={"validation": 88.1, "test": 87.4},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation", "test"]
        )
