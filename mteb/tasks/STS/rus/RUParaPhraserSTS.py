from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class RUParaPhraserSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="RUParaPhraserSTS",
        dataset={
            "path": "merionum/ru_paraphraser",
            "revision": "43265056790b8f7c59e0139acb4be0a8dad2c8f4",
        },
        description="ParaPhraser is a news headlines corpus with precise, near and non-paraphrases.",
        reference="https://aclanthology.org/2020.ngt-1.6",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="cosine_spearman",
        date=("2009-01-01", "2019-01-01"),  # rough estimate,
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{gudkov-etal-2020-automatically,
  address = {Online},
  author = {Gudkov, Vadim  and
Mitrofanova, Olga  and
Filippskikh, Elizaveta},
  booktitle = {Proceedings of the Fourth Workshop on Neural Generation and Translation},
  doi = {10.18653/v1/2020.ngt-1.6},
  month = jul,
  pages = {54--59},
  publisher = {Association for Computational Linguistics},
  title = {Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation},
  url = {https://aclanthology.org/2020.ngt-1.6},
  year = {2020},
}

@inproceedings{pivovarova2017paraphraser,
  author = {Pivovarova, Lidia and Pronoza, Ekaterina and Yagunova, Elena and Pronoza, Anton},
  booktitle = {Conference on artificial intelligence and natural language},
  organization = {Springer},
  pages = {211--225},
  title = {ParaPhraser: Russian paraphrase corpus and shared task},
  year = {2017},
}
""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = -1
        metadata_dict["max_score"] = 1
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {
                "text_1": "sentence1",
                "text_2": "sentence2",
                "class": "score",
            }
        )
        self.dataset = self.dataset.map(lambda x: {"score": float(x["score"])})
