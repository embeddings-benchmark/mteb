from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class STS13VisualSTS(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        name="STS13VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts13",
            "revision": "561ee9ca47ff3e4a657283c59416deca8dc169f2",
        },
        description="SemEval STS 2013 dataset." + "then rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="STS",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-12-31"),
        domains=["Web", "News", "Non-fiction", "Written"],
        task_subtypes=[],
        license="Not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="rendered",
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 1500},
            "avg_character_length": {"dev": 1.0, "test": 1.0},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
