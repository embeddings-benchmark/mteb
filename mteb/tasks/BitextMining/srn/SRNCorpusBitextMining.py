from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "nld_Latn": "nl",
    "srn_Latn": "srn",
}
_SPLIT = ["test"]

# number of sentences to use for evaluation (256 is full test set)
_N = 256

_EVAL_LANGS = {
    "srn_Latn-nld_Latn": ["srn-Latn", "nld-Latn"],
    "nld_Latn-srn_Latn": ["nld-Latn", "srn-Latn"],
}


class SRNCorpusBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="SRNCorpusBitextMining",
        dataset={
            "path": "mteb/SRNCorpusBitextMining",
            "revision": "e0200efcb6654e6d418f9a7c296497fabee8f89d",
        },
        description="SRNCorpus is a machine translation corpus for creole language Sranantongo and Dutch.",
        reference="https://arxiv.org/abs/2212.06383",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLIT,
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=("2022-04-01", "2022-07-31"),
        domains=["Social", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{zwennicker2022towards,
  author = {Zwennicker, Just and Stap, David},
  journal = {arXiv preprint arXiv:2212.06383},
  title = {Towards a general purpose machine translation system for Sranantongo},
  year = {2022},
}
""",
    )
