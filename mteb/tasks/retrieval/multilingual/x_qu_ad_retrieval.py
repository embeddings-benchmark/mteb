from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "ar": ["arb-Arab"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "hi": ["hin-Deva"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "vi": ["vie-Latn"],
    "zh": ["zho-Hans"],
}


class XQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XQuADRetrieval",
        dataset={
            "path": "mteb/XQuADRetrieval",
            "revision": "194842d6a49baddf6cd3f4d5130ab3c7756268d2",
        },
        description="XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.",
        reference="https://huggingface.co/datasets/xquad",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2019-05-21", "2019-11-21"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Artetxe:etal:2019,
  address = {Online},
  author = {Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
  booktitle = {Proceedings of the 58th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics},
  doi = {10.18653/v1/2020.acl-main.421},
  editor = {Jurafsky, Dan and Chai, Joyce and Schluter, Natalie and Tetreault, Joel},
  month = {jul},
  pages = {4623--4637},
  publisher = {Association for Computational Linguistics},
  title = {On the Cross-lingual Transferability of Monolingual Representations},
  url = {https://aclanthology.org/2020.acl-main.421/},
  year = {2020},
}

@inproceedings{dumitrescu2021liro,
  author = {Stefan Daniel Dumitrescu and Petru Rebeja and Beata Lorincz and Mihaela Gaman and Andrei Avram and Mihai Ilie and Andrei Pruteanu and Adriana Stan and Lorena Rosia and Cristina Iacobescu and Luciana Morogan and George Dima and Gabriel Marchidan and Traian Rebedea and Madalina Chitez and Dani Yogatama and Sebastian Ruder and Radu Tudor Ionescu and Razvan Pascanu and Viorica Patraucean},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  title = {LiRo: Benchmark and leaderboard for Romanian language tasks},
  url = {https://openreview.net/forum?id=JH61CD7afTv},
  year = {2021},
}
""",
    )
