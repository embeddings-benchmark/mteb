from mteb.abstasks.image.abs_task_any2any_retrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CMUArcticA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CMUArcticA2TRetrieval",
        description=(
            "Retrieve the correct transcription for an English speech segment. "
            "The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. "
            "The corpora contains 1150 samples based on read-aloud segments from books, which are out of copyright "
            "and derived from the Gutenberg project."
        ),
        reference="http://festvox.org/cmu_arctic/",
        dataset={
            "path": "mteb/CMU_Arctic_a2t",
            "revision": "68e5228b82d03c20c22322ad22008464a32f960b",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2000-01-01", "2002-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{cmu-lti-03-177,
  author = {Clark, Rob and Richmond, Keith},
  institution = {Carnegie Mellon University, Language Technologies Institute},
  number = {CMU-LTI-03-177},
  title = {A detailed report on the CMU Arctic speech database},
  year = {2003},
}
""",
    )


class CMUArcticT2ARetrieval(AbsTaskAny2AnyRetrieval):
    """Text-to-audio retrieval on CMU Arctic transcription ↔ audio pairs."""

    metadata = TaskMetadata(
        name="CMUArcticT2ARetrieval",
        description=(
            "Retrieve the correct audio segment for an English transcription. "
            "The dataset is derived from the phonetically balanced CMU Arctic single-speaker TTS corpora. "
            "The corpora contains 1150 audio-text pairs based on read-aloud segments from public domain books "
            "originally sourced from the Gutenberg project."
        ),
        reference="http://festvox.org/cmu_arctic/",
        dataset={
            "path": "mteb/CMU_Arctic_t2a",
            "revision": "7c845fdfe355c226096203ffd4cdead3229950dc",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2000-01-01", "2002-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@techreport{cmu-lti-03-177,
  author = {Clark, Rob and Richmond, Keith},
  institution = {Carnegie Mellon University, Language Technologies Institute},
  number = {CMU-LTI-03-177},
  title = {A detailed report on the CMU Arctic speech database},
  year = {2003},
}
""",
    )
