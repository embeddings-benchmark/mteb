from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "eng-eng_hin": ["eng-Latn", "hin-Latn"],
}


class PhincBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="PhincBitextMining",
        dataset={
            "path": "gentaiscool/bitext_phinc_miners",
            "revision": "3321d2863453fc96d50e6d861761c323e889f310",
        },
        description="Phinc is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.",
        reference="https://huggingface.co/datasets/veezbo/phinc",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2019-01-01", "2020-01-01"),
        domains=["Social", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{srivastava2020phinc,
  author = {Srivastava, Vivek and Singh, Mayank},
  booktitle = {Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
  pages = {41--49},
  title = {PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation},
  year = {2020},
}
""",
    )
