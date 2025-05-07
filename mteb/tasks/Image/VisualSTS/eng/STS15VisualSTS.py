from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class STS15VisualSTS(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        name="STS15VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts15",
            "revision": "1f8d08d9b9daac7118dfdefeb94b0aac4baf2e5f",
        },
        description="SemEval STS 2015 dataset" + "rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(eng)",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2008-01-01", "2014-07-28"),
        domains=["Blog", "News", "Web", "Written", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="rendered",
        bibtex_citation=r"""
@article{xiao2024pixel,
  author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2402.08183},
  title = {Pixel Sentence Representation Learning},
  year = {2024},
}
""",
        descriptive_stats={
            "n_samples": {"test": 3000},
            "avg_character_length": {"dev": 1.0, "test": 1.0},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
