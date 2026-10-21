from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LibriTTSA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LibriTTSA2TRetrieval",
        description=(
            "Given audiobook speech segments from the multi‑speaker LibriTTS corpus, "
            "retrieve the correct text transcription. LibriTTS is a 585‑hour, 24 kHz, "
            "multi‑speaker English TTS corpus derived from LibriVox (audio) and Project Gutenberg (text)."
        ),
        reference="https://www.openslr.org/60/",
        dataset={
            "path": "mteb/LibriTTS_a2t",
            "revision": "dbf3f317f96023e103b98548a3b99cfa919afb56",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2019-11-01", "2019-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        superseded_by="LibriTTSA2TRetrieval.v2",
        bibtex_citation=r"""
@misc{zen2019librittscorpusderivedlibrispeech,
  archiveprefix = {arXiv},
  author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
  eprint = {1904.02882},
  primaryclass = {cs.SD},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://arxiv.org/abs/1904.02882},
  year = {2019},
}
""",
    )


class LibriTTST2ARetrieval(AbsTaskRetrieval):
    """Retrieval of speech segments given text queries, on the LibriTTS dataset."""

    metadata = TaskMetadata(
        name="LibriTTST2ARetrieval",
        description=(
            "Given an English text transcription, retrieve its corresponding audiobook "
            "speech segment from the multi‑speaker LibriTTS corpus. LibriTTS is a 585‑hour, 24 kHz, "
            "multi‑speaker English TTS corpus derived from LibriVox and Project Gutenberg."
        ),
        reference="https://www.openslr.org/60/",
        dataset={
            "path": "mteb/LibriTTS_t2a",
            "revision": "d6fc5fdcdc0940892c84461e8cfcb908baa717e4",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2019-11-01", "2019-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        superseded_by="LibriTTST2ARetrieval.v2",
        bibtex_citation=r"""
@misc{zen2019librittscorpusderivedlibrispeech,
  archiveprefix = {arXiv},
  author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
  eprint = {1904.02882},
  primaryclass = {cs.SD},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://arxiv.org/abs/1904.02882},
  year = {2019},
}
""",
    )


class LibriTTSA2TRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LibriTTSA2TRetrieval.v2",
        description=(
            "Given audiobook speech segments from the multi‑speaker LibriTTS corpus, "
            "retrieve the correct text transcription. LibriTTS is a 585‑hour, 24 kHz, "
            "multi‑speaker English TTS corpus derived from LibriVox (audio) and Project Gutenberg (text)."
            " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less."
        ),
        reference="https://www.openslr.org/60/",
        dataset={
            "path": "mteb/LibriTTS_a2t",
            "revision": "4c5387cc32913b5b480522d23422a332b652ee2a",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2019-11-01", "2019-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["LibriTTSA2TRetrieval"],
        bibtex_citation=r"""
@misc{zen2019librittscorpusderivedlibrispeech,
  archiveprefix = {arXiv},
  author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
  eprint = {1904.02882},
  primaryclass = {cs.SD},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://arxiv.org/abs/1904.02882},
  year = {2019},
}
""",
    )


class LibriTTST2ARetrievalV2(AbsTaskRetrieval):
    """Retrieval of speech segments given text queries, on the LibriTTS dataset."""

    metadata = TaskMetadata(
        name="LibriTTST2ARetrieval.v2",
        description=(
            "Given an English text transcription, retrieve its corresponding audiobook "
            "speech segment from the multi‑speaker LibriTTS corpus. LibriTTS is a 585‑hour, 24 kHz, "
            "multi‑speaker English TTS corpus derived from LibriVox and Project Gutenberg."
            " Version 2 merges queries and documents with identical content, so a repeated query counts every matching item as relevant, and drops texts of 3 characters or less."
        ),
        reference="https://www.openslr.org/60/",
        dataset={
            "path": "mteb/LibriTTS_t2a",
            "revision": "70a5685ab4eeacf47c56763ae20300987e2dc6a1",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2019-11-01", "2019-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["LibriTTST2ARetrieval"],
        bibtex_citation=r"""
@misc{zen2019librittscorpusderivedlibrispeech,
  archiveprefix = {arXiv},
  author = {Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
  eprint = {1904.02882},
  primaryclass = {cs.SD},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://arxiv.org/abs/1904.02882},
  year = {2019},
}
""",
    )
