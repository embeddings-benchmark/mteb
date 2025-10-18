from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class PawsXPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsXPairClassification",
        dataset={
            "path": "mteb/PawsXPairClassification",
            "revision": "558da352e7dba3ed3229fc4922aef2ebaff0a90b",
        },
        description="{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
        reference="https://arxiv.org/abs/1908.11828",
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
@misc{yang2019pawsx,
  archiveprefix = {arXiv},
  author = {Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
  eprint = {1908.11828},
  primaryclass = {cs.CL},
  title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
  year = {2019},
}
""",
    )
