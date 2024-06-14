from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

_EVAL_SPLIT = "test"


class STSIDMT(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSIDMT",
        dataset={
            "path": "LazarusNLP/stsb_mt_id",
            "revision": "d0095a14faeaad923263a0ce2d301f8163b94106",
            "trust_remote_code": True,
        },
        description="Indonesian test sets from STS-B translated using Google Translate API",
        reference="https://huggingface.co/datasets/LazarusNLP/stsb_mt_id",
        type="STS",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["ind-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@software{wongso_2024_10983756,
  author       = {Wongso, Wilson and
                  Joyoadikusumo, Ananto and
                  Setiawan, David Samuel and
                  Limcorn, Steven},
  title        = {LazarusNLP/indonesian-sentence-embeddings: v0.0.1},
  month        = apr,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.10983756},
  url          = {https://doi.org/10.5281/zenodo.10983756}
}
""",
        n_samples={"test":1380},
        avg_character_length={"test":3.45},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        self.dataset = self.dataset.remove_columns("score")
        type_cast = self.dataset.features.copy()
        type_cast["correlation"] = Value("int64")
        self.dataset = self.dataset.cast(new_features)
        self.dataset = self.dataset.rename_column("correlation", "score")
        self.dataset = self.dataset.rename_column("text_1", "sentence1")
        self.dataset = self.dataset.rename_column("text_2", "sentence2")
