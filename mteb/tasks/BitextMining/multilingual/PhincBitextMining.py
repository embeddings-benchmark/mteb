from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

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
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2019-01-01", "2020-01-01"),
        form=["written"],
        domains=["Social"],
        task_subtypes=[],
        license="CC BY 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{srivastava2020phinc,
        title={PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine Translation},
        author={Srivastava, Vivek and Singh, Mayank},
        booktitle={Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
        pages={41--49},
        year={2020}
        }
        """,
        n_samples={"train": 13738},
        avg_character_length={"train": 75.32},
    )
