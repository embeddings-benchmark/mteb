from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class ArabicSTSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ArabicSTSBenchmarkSTS",
        dataset={
            "path": "Ruqiya/stsbenchmark-sts-ar",
            "revision": "44228871f6671288a0580db6e7e12831b4d16d3b",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Arabic. "
        "Translations were originally done by Helsinki-NLP/opus-mt-en-ar model",
        reference="https://huggingface.co/datasets/Ruqiya/stsbenchmark-sts-ar",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["ara-Arab"],
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
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
