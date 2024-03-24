from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

_EVAL_SPLIT = "test"


class STSES(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSES",
        hf_hub_name="PlanTL-GOB-ES/sts-es",
        description="Spanish test sets from SemEval-2014 (Agirre et al., 2014) and SemEval-2015 (Agirre et al., 2015)",
        reference="https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es",
        type="STS",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["es"],
        main_score="cosine_spearman",
        revision="0912bb6c9393c76d62a7c5ee81c4c817ff47c9f4",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return dict(self.metadata)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = load_dataset(
            self.metadata_dict["hf_hub_name"],
            trust_remote_code=True,
            revision=self.metadata_dict.get("revision", None),
        )[_EVAL_SPLIT]
        data = data.add_column("score", [d["label"] for d in data])
        self.dataset = {_EVAL_SPLIT: data}

        self.data_loaded = True
