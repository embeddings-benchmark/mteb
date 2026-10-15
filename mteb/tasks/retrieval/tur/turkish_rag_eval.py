from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishRAGEvalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TurkishRAGEvalRetrieval",
        dataset={
            "path": "mteb/turkish-rag-eval",
            "revision": "1666040a6b9b383e358b0ec34ec40958c619b507",
        },
        description="Hand-written Turkish questions over Turkish Wikipedia health articles; each question is labelled with the Wikipedia passage (at most 700 characters, split on section headings) that contains its answer span.",
        reference="https://github.com/RizgarOzan/turkish-rag-eval",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"passages": ["tur-Latn"]},
        main_score="ndcg_at_10",
        date=("2026-09-01", "2026-09-19"),
        domains=["Encyclopaedic", "Medical", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )
