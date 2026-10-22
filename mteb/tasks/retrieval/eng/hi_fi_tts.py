from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HiFiTTSA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HiFiTTSA2TRetrieval",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_a2t",
            "revision": "4535eb399017ce24ce41cfcdc7c157bf76bcfa27",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        superseded_by="HiFiTTSA2TRetrieval.v2",
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )


class HiFiTTST2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HiFiTTST2ARetrieval",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_t2a",
            "revision": "c5783d12999ec5f6a9f337990cfd46cc131e2b0f",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        superseded_by="HiFiTTST2ARetrieval.v2",
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )


class HiFiTTSA2TRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HiFiTTSA2TRetrieval.v2",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg."
        " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_a2t",
            "revision": "143bec4e5a842800a5531fccdde8f13dbc2ac6d9",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["HiFiTTSA2TRetrieval"],
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )


class HiFiTTST2ARetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HiFiTTST2ARetrieval.v2",
        description="Sentence-level text captions aligned to 44.1 kHz audiobook speech segments from the Hi‑Fi Multi‑Speaker English TTS dataset. Dataset is based on public audiobooks from LibriVox and texts from Project Gutenberg."
        " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less.",
        reference="https://openslr.org/109/",
        dataset={
            "path": "mteb/hifi-tts_t2a",
            "revision": "ffef62ca97170c1a61cb53486c0d441fe07daa40",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["HiFiTTST2ARetrieval"],
        bibtex_citation=r"""
@article{bakhturina2021hi,
  author = {Bakhturina, Evelina and Lavrukhin, Vitaly and Ginsburg, Boris and Zhang, Yang},
  journal = {arXiv preprint arXiv:2104.01497},
  title = {{Hi-Fi Multi-Speaker English TTS Dataset}},
  year = {2021},
}
""",
    )
