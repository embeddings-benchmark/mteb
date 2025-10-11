from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

TEST_SAMPLES = 2048


class VieMedEVBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="VieMedEVBitextMining",
        dataset={
            "path": "mteb/VieMedEVBitextMining",
            "revision": "56e8a74cdafa10aaceb9fec8272c209b800165de",
        },
        description="A high-quality Vietnamese-English parallel data from the medical domain for machine translation",
        reference="https://aclanthology.org/2015.iwslt-evaluation.11/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "vie-Latn"],
        main_score="f1",
        date=("2024-08-28", "2022-03-28"),
        domains=["Medical", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",  # version is assumed, but was previously unspecified
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@inproceedings{medev,
  author = {Nhu Vo and Dat Quoc Nguyen and Dung D. Le and Massimo Piccardi and Wray Buntine},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)},
  title = {{Improving Vietnamese-English Medical Machine Translation}},
  year = {2024},
}
""",
    )
