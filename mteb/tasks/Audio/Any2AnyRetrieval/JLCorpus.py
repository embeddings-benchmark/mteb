from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class JLCorpusA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="JLCorpusA2TRetrieval",
        description=(
            "Emotional speech segments from the JL-Corpus, "
            "balanced over long vowels and annotated for primary and secondary emotions."
        ),
        reference="https://www.kaggle.com/tli725/jl-corpus",
        dataset={
            "path": "mteb/JL-Corpus_a2t",
            "revision": "13eb64cc59448af4879122a560fe20234945cae7",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{james2018open,
  author = {James, Jesin and Li, Tian and Watson, Catherine},
  booktitle = {Proc. Interspeech 2018},
  title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
  year = {2018},
}
""",
    )


class JLCorpusT2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Text-to-audio retrieval on JL-Corpus emotional speech captions ↔ audio pairs."""

    metadata = TaskMetadata(
        name="JLCorpusT2ARetrieval",
        description=(
            "Emotional speech segments from the JL-Corpus, "
            "balanced over long vowels and annotated for primary and secondary emotions."
        ),
        reference="https://www.kaggle.com/tli725/jl-corpus",
        dataset={
            "path": "mteb/JL-Corpus_t2a",
            "revision": "65e3ea8c3077fabbea5d46f8783b5e440031aba1",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{james2018open,
  author = {James, Jesin and Li, Tian and Watson, Catherine},
  booktitle = {Proc. Interspeech 2018},
  title = {An Open Source Emotional Speech Corpus for Human Robot Interaction Applications},
  year = {2018},
}
""",
    )
