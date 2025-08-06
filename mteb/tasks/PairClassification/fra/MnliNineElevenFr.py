from datasets import load_dataset
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MnliNineElevenFrTask(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="mnli-nineeleven-fr-mt",
        description=(
            "Natural Language Inference on the MNLI NineEleven dataset "
            "translated to French: predict the relation between two sentences "
            "(0=entailment, 1=neutral, 2=contradiction)."
        ),
        reference="https://huggingface.co/datasets/maximoss/mnli-nineeleven-fr-mt",
        dataset={
            "path": "maximoss/mnli-nineeleven-fr-mt",
            "revision": "b434985928ce6ad5c02703028323f3a5744eb25b",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["fra-Latn"],
        main_score="max_accuracy",
        date=("2025-08-05", "2025-08-05"),
        domains=["News"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{N18-1101,
  author    = {Williams, Adina and Nangia, Nikita and Bowman, Samuel},
  title     = {A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
  booktitle = {Proceedings of the 2018 NAACL-HLT, Volume 1 (Long Papers)},
  year      = {2018},
  publisher = {ACL},
  pages     = {1112--1122},
  location  = {New Orleans, Louisiana},
  url       = {http://aclweb.org/anthology/N18-1101},
}
""",
    )

    def dataset_transform(self):

        out: dict[str, dict[str, list[dict[str, list]]]] = {}
        for lang in self.hf_subsets:
            out[lang] = {}
            for split in self.metadata.eval_splits:
                ds = self.dataset[split]
                ds = ds.map(lambda ex: {"label": 1 if ex["label"] == "1" else 0})
                out[lang][split] = [
                    {
                        "sentence1": ds["premise"],
                        "sentence2": ds["hypothesis"],
                        "labels": ds["label"],
                    }
                ]
        self.dataset = out