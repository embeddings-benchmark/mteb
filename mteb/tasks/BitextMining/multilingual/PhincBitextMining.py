from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "eng-eng_hin": ["eng-Latn", "hin-Latn"],
}


class PhincBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="PhincBitextMining",
        dataset={
            "path": "gentaiscool/bitext_phinc_miners",
            "revision": "3321d2863453fc96d50e6d861761c323e889f310",
        },
        description="Phinc is a parallel corpus for machine translation pairing code-mixed Hinglish (a fusion of Hindi and English commonly used in modern India) with human-generated English translations.",
        reference="https://huggingface.co/datasets/veezbo/phinc",
        type="BitextMining",
        category="s2s",
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
        bibtex_citation="""
        @inproceedings{srivastava2020phinc,
        title={PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation},
        author={Srivastava, Vivek and Singh, Mayank},
        booktitle={Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
        pages={41--49},
        year={2020}
        }
        """,
        descriptive_stats={
            "n_samples": {"train": 13738},
            "avg_character_length": {"train": 75.32},
        },
    )
