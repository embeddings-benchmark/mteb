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
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="cosine_spearman",
        date=("2009-01-01", "2019-01-01"),  # rough estimate,
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{gudkov-etal-2020-automatically,
          title = "Automatically Ranked {R}ussian Paraphrase Corpus for Text Generation",
          author = "Gudkov, Vadim  and
            Mitrofanova, Olga  and
            Filippskikh, Elizaveta",
          booktitle = "Proceedings of the Fourth Workshop on Neural Generation and Translation",
          month = jul,
          year = "2020",
          address = "Online",
          publisher = "Association for Computational Linguistics",
          url = "https://aclanthology.org/2020.ngt-1.6",
          doi = "10.18653/v1/2020.ngt-1.6",
          pages = "54--59",
        }
        @inproceedings{pivovarova2017paraphraser,
          title={ParaPhraser: Russian paraphrase corpus and shared task},
          author={Pivovarova, Lidia and Pronoza, Ekaterina and Yagunova, Elena and Pronoza, Anton},
          booktitle={Conference on artificial intelligence and natural language},
          pages={211--225},
          year={2017},
          organization={Springer}
        }
        """,
        n_samples={"test": 1924},
        avg_character_length={"test": 61.25},
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
