from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class AudioSetStrongA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="AudioSetStrongA2TRetrieval",
        description=(
            "Retrieve all temporally-strong labeled events within 10s audio clips "
            "from the AudioSet Strongly-Labeled subset."
        ),
        reference="https://research.google.com/audioset/download_strong.html",
        dataset={
            "path": "mteb/audioset_strong_a2t",
            "revision": "bca12edd4bb25d19cd5f09574e81ea202aacf4bd",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2021-05-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{hershey2021benefittemporallystronglabelsaudio,
  archiveprefix = {arXiv},
  author = {Shawn Hershey and Daniel P W Ellis and Eduardo Fonseca and Aren Jansen and Caroline Liu and R Channing Moore and Manoj Plakal},
  eprint = {2105.07031},
  primaryclass = {cs.SD},
  title = {The Benefit Of Temporally-Strong Labels In Audio Event Classification},
  url = {https://arxiv.org/abs/2105.07031},
  year = {2021},
}
""",
    )


class AudioSetStrongT2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Text-to-audio retrieval on AudioSet Strong labels ↔ audio segments."""

    metadata = TaskMetadata(
        name="AudioSetStrongT2ARetrieval",
        description=(
            "Retrieve audio segments corresponding to a given sound event label "
            "from the AudioSet Strongly-Labeled 10s clips."
        ),
        reference="https://research.google.com/audioset/download_strong.html",
        dataset={
            "path": "mteb/audioset_strong_t2a",
            "revision": "d4a2fc5c25533a57634eadc00267d694e23aa00d",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2021-05-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{hershey2021benefittemporallystronglabelsaudio,
  archiveprefix = {arXiv},
  author = {Shawn Hershey and Daniel P W Ellis and Eduardo Fonseca and Aren Jansen and Caroline Liu and R Channing Moore and Manoj Plakal},
  eprint = {2105.07031},
  primaryclass = {cs.SD},
  title = {The Benefit Of Temporally-Strong Labels In Audio Event Classification},
  url = {https://arxiv.org/abs/2105.07031},
  year = {2021},
}
""",
    )
