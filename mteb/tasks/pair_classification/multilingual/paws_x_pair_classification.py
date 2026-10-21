from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PawsXPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsXPairClassification",
        dataset={
            "path": "mteb/PawsXPairClassification",
            "revision": "558da352e7dba3ed3229fc4922aef2ebaff0a90b",
        },
        description="PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
        reference="https://doi.org/10.18653/v1/d19-1382",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test", "validation"],
        eval_langs={
            "de": ["deu-Latn"],
            "en": ["eng-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Hira"],
            "ko": ["kor-Hang"],
            "zh": ["cmn-Hans"],
        },
        main_score="max_ap",
        date=("2016-01-01", "2018-12-31"),
        domains=["Web", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="https://huggingface.co/datasets/google-research-datasets/paws-x#licensing-information",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{yang2019pawsx,
  author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  doi = {10.18653/v1/d19-1382},
  pages = {3685–3690},
  publisher = {Association for Computational Linguistics},
  title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
  url = {https://doi.org/10.18653/v1/d19-1382},
  year = {2019},
}
""",
    )
