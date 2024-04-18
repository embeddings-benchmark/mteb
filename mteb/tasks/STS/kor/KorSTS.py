from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class KorSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorSTS",
        dataset={
            "path": "dkoterwa/kor-sts",
            "revision": "016f35f9b961daaaa7a352e927084e3da662ac1f",
        },
        description="",
        reference="https://arxiv.org/abs/2004.03289",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="CC-BY-SA-4.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}""",
        n_samples={"test": 1379},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
