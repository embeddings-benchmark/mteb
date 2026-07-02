from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakSumSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SlovakSumSTS",
        dataset={
            "path": "slovak-nlp/slovak-sts-synthetic",
            "revision": "6286d808151adb399ef35b5bfaba1d7461ee6df3",
        },
        description="The dataset consists of sentence pairs for semantic textual similarity (STS) scoring in Slovak. The pairs were generated using text from the SlovakSum dataset, where an LLM was used to create corresponding sentence pairs for each STS score (0-5). The pairs in the test split were verified by human annotators to ensure quality.",
        reference="https://huggingface.co/datasets/slovak-nlp/slovak-sts-synthetic",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="cosine_spearman",
        date=("2025-11-01", "2025-12-15"),
        domains=["News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=r"""""",
    )

    min_score = 0
    max_score = 5

    def dataset_transform(self):
        _dataset = self.dataset.rename_columns({"similarity_score": "score"})

        # ensure numeric value
        _dataset = _dataset.map(lambda example: {"score": float(example["score"])})

        self.dataset = _dataset
