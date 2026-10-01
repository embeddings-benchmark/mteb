from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "cs": ["ces-Latn"],
    "de": ["deu-Latn"],
    "fr": ["fra-Latn"],
}

_BIBTEX = r"""
@inproceedings{barrault2018findings,
  address = {Belgium, Brussels},
  author = {Barrault, Lo{\"i}c and Bougares, Fethi and Specia, Lucia and Lala, Chiraag and Elliott, Desmond and Frank, Stella},
  booktitle = {Proceedings of the Third Conference on Machine Translation: Shared Task Papers},
  pages = {304--323},
  publisher = {Association for Computational Linguistics},
  title = {Findings of the Third Shared Task on Multimodal Machine Translation},
  year = {2018},
}

@inproceedings{elliott-etal-2017-findings,
  address = {Copenhagen, Denmark},
  author = {Elliott, Desmond and Frank, Stella and Barrault, Lo{\"i}c and Bougares, Fethi and Specia, Lucia},
  booktitle = {Proceedings of the Second Conference on Machine Translation},
  doi = {10.18653/v1/W17-4718},
  month = sep,
  pages = {215--233},
  publisher = {Association for Computational Linguistics},
  title = {Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description},
  url = {https://aclanthology.org/W17-4718/},
  year = {2017},
}

@inproceedings{elliott2016multi30k,
  address = {Berlin, Germany},
  author = {Elliott, Desmond and Frank, Stella and Sima'an, Khalil and Specia, Lucia},
  booktitle = {Proceedings of the 5th Workshop on Vision and Language},
  pages = {70--74},
  publisher = {Association for Computational Linguistics},
  title = {Multi30K: Multilingual English-German Image Descriptions},
  year = {2016},
}
"""

_DESCRIPTION_TAIL = (
    "The Czech, German and French captions are independent human translations describing "
    "the same Flickr30k image rather than machine translations, so the four language "
    "subsets are directly comparable. Built from the 1,000-image test split of "
    "romrawinjp/multi30k; the image side is identical across subsets and is stored once "
    "rather than duplicated per language. Construction script: "
    "scripts/data/multi30k_retrieval/create_data.py."
)


class Multi30kT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Multi30kT2IRetrieval",
        description=(
            "Retrieve Flickr30k images from parallel English, Czech, German and French "
            "descriptions of the same image. " + _DESCRIPTION_TAIL
        ),
        reference="https://aclanthology.org/W16-3210/",
        dataset={
            "path": "vnahata/Multi30k-T2I",
            "revision": "b41de7a143914c34c751b60f139fa8b5d7cd5211",
        },
        type="Any2AnyMultilingualRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2018-12-31"),
        domains=["Scene", "Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )


class Multi30kI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Multi30kI2TRetrieval",
        description=(
            "Retrieve English, Czech, German and French descriptions from a Flickr30k "
            "image, testing cross-lingual image-to-text alignment. " + _DESCRIPTION_TAIL
        ),
        reference="https://aclanthology.org/W16-3210/",
        dataset={
            "path": "vnahata/Multi30k-I2T",
            "revision": "43493b9dd88a89493b0f5904a4118012b5b66acf",
        },
        type="Any2AnyMultilingualRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2018-12-31"),
        domains=["Scene", "Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )
