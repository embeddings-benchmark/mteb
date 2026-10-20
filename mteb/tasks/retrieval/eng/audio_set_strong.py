from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class AudioSetStrongA2TRetrieval(AbsTaskRetrieval):
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2021-05-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hershey2021benefittemporallystronglabelsaudio,
  author = {Hershey, Shawn and Ellis, Daniel P W and Fonseca, Eduardo and Jansen, Aren and Liu, Caroline and Channing Moore, R and Plakal, Manoj},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/icassp39728.2021.9414579},
  month = {June},
  pages = {366–370},
  publisher = {IEEE},
  title = {The Benefit of Temporally-Strong Labels in Audio Event Classification},
  url = {https://doi.org/10.1109/icassp39728.2021.9414579},
  year = {2021},
}
""",
    )


class AudioSetStrongT2ARetrieval(AbsTaskRetrieval):
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2021-05-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hershey2021benefittemporallystronglabelsaudio,
  author = {Hershey, Shawn and Ellis, Daniel P W and Fonseca, Eduardo and Jansen, Aren and Liu, Caroline and Channing Moore, R and Plakal, Manoj},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/icassp39728.2021.9414579},
  month = {June},
  pages = {366–370},
  publisher = {IEEE},
  title = {The Benefit of Temporally-Strong Labels in Audio Event Classification},
  url = {https://doi.org/10.1109/icassp39728.2021.9414579},
  year = {2021},
}
""",
    )
