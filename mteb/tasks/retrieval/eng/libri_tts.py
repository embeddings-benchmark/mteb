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
        bibtex_citation=r"""
@inproceedings{zen2019librittscorpusderivedlibrispeech,
  author = {Zen, Heiga and Dang, Viet and Clark, Rob and Zhang, Yu and Weiss, Ron J. and Jia, Ye and Chen, Zhifeng and Wu, Yonghui},
  booktitle = {Interspeech 2019},
  collection = {interspeech_2019},
  doi = {10.21437/interspeech.2019-2441},
  month = {Sept},
  pages = {1526–1530},
  publisher = {ISCA},
  series = {interspeech_2019},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://doi.org/10.21437/interspeech.2019-2441},
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
        bibtex_citation=r"""
@inproceedings{zen2019librittscorpusderivedlibrispeech,
  author = {Zen, Heiga and Dang, Viet and Clark, Rob and Zhang, Yu and Weiss, Ron J. and Jia, Ye and Chen, Zhifeng and Wu, Yonghui},
  booktitle = {Interspeech 2019},
  collection = {interspeech_2019},
  doi = {10.21437/interspeech.2019-2441},
  month = {Sept},
  pages = {1526–1530},
  publisher = {ISCA},
  series = {interspeech_2019},
  title = {LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech},
  url = {https://doi.org/10.21437/interspeech.2019-2441},
  year = {2019},
}
""",
    )
