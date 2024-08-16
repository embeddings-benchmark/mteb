from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

_EVAL_SPLIT = "test"


class STSES(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSES",
        dataset={
            "path": "PlanTL-GOB-ES/sts-es",
            "revision": "0912bb6c9393c76d62a7c5ee81c4c817ff47c9f4",
            "trust_remote_code": True,
        },
        description="Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)",
        reference="https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["spa-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Written"],
        task_subtypes=None,
        license="cc-by-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{agirre2015semeval,
  title={Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability},
  author={Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel and Diab, Mona and Gonzalez-Agirre, Aitor and Guo, Weiwei and Lopez-Gazpio, Inigo and Maritxalar, Montse and Mihalcea, Rada and others},
  booktitle={Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015)},
  pages={252--263},
  year={2015}
}


@inproceedings{agirre2014semeval,
  title={SemEval-2014 Task 10: Multilingual Semantic Textual Similarity.},
  author={Agirre, Eneko and Banea, Carmen and Cardie, Claire and Cer, Daniel M and Diab, Mona T and Gonzalez-Agirre, Aitor and Guo, Weiwei and Mihalcea, Rada and Rigau, German and Wiebe, Janyce},
  booktitle={SemEval@ COLING},
  pages={81--91},
  year={2014}
}
""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self):
        data = self.dataset[_EVAL_SPLIT]
        data = data.add_column("score", [d["label"] for d in data])
        self.dataset = {_EVAL_SPLIT: data}
