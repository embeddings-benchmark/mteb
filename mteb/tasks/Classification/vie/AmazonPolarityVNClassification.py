from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class AmazonPolarityVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityVNClassification",
        description="Amazon Polarity Classification Dataset.",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "GreenNode/amazon-polarity-vn",
            "revision": "4e9a0d6e6bd97ab32f23c50c043d751eed2a5f8a"
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        form=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        socioeconomic_status=None,
        text_creation=None,
        bibtex_citation="""
@misc{pham2025vnmtebvietnamesemassivetext,
    title={VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
    author={Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
    year={2025},
    eprint={2507.21500},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.21500}
}
""",
        adapted_from=["AmazonPolarityClassification"],
    )
