from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishRAGEvalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TurkishRAGEvalRetrieval",
        dataset={
            "path": "RizgarOzan/turkish-rag-eval",
            "revision": "820ae99ad4425198de503ad2a1434c8bdfce6749",
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
        bibtex_citation=r"""
@misc{ozan2026turkishrageval,
  author = {Ozan, Rızgar},
  howpublished = {\url{https://huggingface.co/datasets/RizgarOzan/turkish-rag-eval}},
  title = {Turkish RAG Eval: a retrieval test set for Turkish Wikipedia},
  year = {2026},
}
""",
    )
