from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakPharmacyDrMaxReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakPharmacyDrMaxReranking",
        description=(
            "A reranking dataset created from Q&A content collected from DrMax pharmacy website. "
            "The dataset consists of questions about medications, health conditions, and pharmaceutical advice, "
            "with answers provided by qualified pharmacists. This dataset is designed to evaluate models' "
            "ability to rank relevant pharmaceutical information and expert responses."
        ),
        reference="https://huggingface.co/datasets/kinit/slovak-pharmacy-drmax-reranking",
        dataset={
            "path": "slovak-nlp/slovak-pharmacy-drmax-reranking",
            "revision": "b557bbc99cf4d410b32a5e25a22f34f08b540e06",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-11-01", "2025-11-30"),
        domains=[
            "Medical",
            "Web",
        ],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""""",
    )


class SlovakPharmacyMojaLekarenReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakPharmacyMojaLekarenReranking",
        description=(
            "A reranking dataset created from Q&A content collected from MojaLekaren pharmacy website. "
            "The dataset consists of questions about medications, health conditions, and pharmaceutical advice, "
            "with answers provided by qualified pharmacists. This dataset is designed to evaluate models' "
            "ability to rank relevant pharmaceutical information and expert responses."
        ),
        reference="https://huggingface.co/datasets/slovak-nlp/slovak-pharmacy-mojalekaren-reranking",
        dataset={
            "path": "slovak-nlp/slovak-pharmacy-mojalekaren-reranking",
            "revision": "e4ac8bb50c5252d154996df080d9901774556368",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-11-01", "2025-11-30"),
        domains=[
            "Medical",
            "Web",
        ],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""""",
    )
