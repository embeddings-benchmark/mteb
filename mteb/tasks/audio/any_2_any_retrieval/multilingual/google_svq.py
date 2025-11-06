from mteb.abstasks.image.abs_task_any2any_retrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


# Google SVQ supports 17 languages
_EVAL_LANGS = {
    "ar_eg": ["arz"],
    "ar_x_gulf": ["acm"],
    "ar_x_levant": ["apc"],
    "ar_x_maghrebi": ["arq"],
    "bn_bd": ["ben"],
    "bn_in": ["ben"],
    "en_au": ["eng"],
    "en_gb": ["eng"],
    "en_in": ["eng"],
    "en_ph": ["eng"],
    "en_us": ["eng"],
    "fi_fi": ["fin"],
    "id_id": ["ind"],
    "ko_kr": ["kor"],
    "ru_ru": ["rus"],
    "sw": ["swa"],
    "te_in": ["tel"]
}

class GoogleSVQA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="GoogleSVQA2TRetrieval",
        description="Natural language transcription for short voice questions",
        reference="https://github.com/nateraw/download-musiccaps-dataset",
        dataset={
            "path": "google/svq",
            "revision": "177e4fa88e59148dc746471e164b0b46b193f41f",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2025-01-01", "2025-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
        @inproceedings{
        heigold2025massive,
        title={Massive Sound Embedding Benchmark ({MSEB})},
        author={Georg Heigold and Ehsan Variani and Tom Bagby and Cyril Allauzen and Ji Ma and Shankar Kumar and Michael Riley},
        booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
        year={2025},
        url={https://openreview.net/forum?id=X0juYgFVng}
        }
        """,
    )


