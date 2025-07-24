from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CommonVoiceA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceA2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ardila2019common,
  title={Common voice: A massively-multilingual speech corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}
""",
    )


class CommonVoiceT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceT2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ardila2019common,
  title={Common voice: A massively-multilingual speech corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}
""",
    )
