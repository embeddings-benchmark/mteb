from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

_EVAL_SPLIT = "test"


class STSMTSV(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSMTSV",
        dataset={
            "path": "timpal0l/stsb_mt_sv",
            "revision": "ed6ac3f11354fadbc1d23d44b737fce3c889ce50",
            "trust_remote_code": True,
        },
        # description="Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)",
        description="STSbenchmark English dataset translated using Google machine translation API to swedish.",
        reference="https://huggingface.co/datasets/timpal0l/stsb_mt_sv",
        type="STS",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["swe-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2020-9-7"),
        form=["spoken", "written"],
        domains=["News", "Social", "Web"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="machine-translated",
        bibtex_citation="""@misc{isbister2020simply,
      title={Why Not Simply Translate? A First Swedish Evaluation Benchmark for Semantic Similarity}, 
      author={Tim Isbister and Magnus Sahlgren},
      year={2020},
      eprint={2009.03116},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
        }""",
        n_samples={"train": 5749, "test": 1379,'validation':1500},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    # def dataset_transform(self):
    #     data = self.dataset[_EVAL_SPLIT]
    #     data = data.add_column("score", [d["label"] for d in data])
    #     self.dataset = {_EVAL_SPLIT: data}
