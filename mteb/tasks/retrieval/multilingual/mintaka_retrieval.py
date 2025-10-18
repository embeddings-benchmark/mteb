from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"
_LANGS = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Hira"],
    "pt": ["por-Latn"],
}


class MintakaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MintakaRetrieval",
        description="We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. ",
        reference=None,
        dataset={
            "path": "mteb/MintakaRetrieval",
            "revision": "43bc699486e768138ce3a6d4cd859da306ac9eef",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-01-01"),  # best guess: based on the date of the paper
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",  # best guess
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{sen-etal-2022-mintaka,
  address = {Gyeongju, Republic of Korea},
  author = {Sen, Priyanka  and
Aji, Alham Fikri  and
Saffari, Amir},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  month = oct,
  pages = {1604--1619},
  publisher = {International Committee on Computational Linguistics},
  title = {Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering},
  url = {https://aclanthology.org/2022.coling-1.138},
  year = {2022},
}
""",
    )
