from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "as": ["asm-Beng"],
    "bn": ["ben-Beng"],
    "gu": ["guj-Gujr"],
    "hi": ["hin-Deva"],
    "kn": ["kan-Knda"],
    "ml": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "or": ["ory-Orya"],
    "pa": ["pan-Guru"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
}


class IndicXnliPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="IndicXnliPairClassification",
        dataset={
            "path": "mteb/IndicXnliPairClassification",
            "revision": "027e97b9afe84ea3447b57b7705b8864bb2b3a83",
        },
        description=(
            "INDICXNLI is similar to existing XNLI dataset in shape/form, but "
            "focuses on Indic language family. "
            "The train (392,702), validation (2,490), and evaluation sets (5,010) of English "
            "XNLI were translated from English into each of the eleven Indic languages. IndicTrans "
            "is a large Transformer-based sequence to sequence model. It is trained on Samanantar "
            "dataset (Ramesh et al., 2021), which is the largest parallel multi- lingual corpus "
            "over eleven Indic languages."
        ),
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="max_ap",
        date=("2022-04-22", "2022-10-06"),
        domains=["Non-fiction", "Fiction", "Government", "Written"],
        task_subtypes=None,
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{aggarwal_gupta_kunch_22,
  author = {Aggarwal, Divyanshu and Gupta, Vivek and Kunchukuttan, Anoop},
  copyright = {Creative Commons Attribution 4.0 International},
  doi = {10.48550/ARXIV.2204.08776},
  publisher = {arXiv},
  title = {IndicXNLI: Evaluating Multilingual Inference for Indian Languages},
  url = {https://arxiv.org/abs/2204.08776},
  year = {2022},
}
""",
    )
