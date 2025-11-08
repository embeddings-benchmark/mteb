from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AtisIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AtisIntentClassification",
        description="The ATIS Spoken Language Systems Pilot Corpus",
        dataset={
            "path": "DeepPavlov/atis_intent_classification",
            "revision": "01f3bc14cd27fdd6d4c4ac07e30e6ad3fa09c0a6",
        },
        reference="https://huggingface.co/datasets/fathyshalab/atis_intents",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1990-01-01", "1990-01-01"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{hemphill-etal-1990-atis,
            title = "The {ATIS} Spoken Language Systems Pilot Corpus",
            author = "Hemphill, Charles T.  and
              Godfrey, John J.  and
              Doddington, George R.",
            booktitle = "Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, {P}ennsylvania, June 24-27,1990",
            year = "1990",
            url = "https://aclanthology.org/H90-1021/"
        }""",
    )
