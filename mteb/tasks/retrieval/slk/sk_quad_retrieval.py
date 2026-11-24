import logging

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class SKQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SKQuadRetrieval",
        description=(
            "Retrieval SK Quad evaluates Slovak search performance using questions and answers "
            "derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers "
            "based on their relevancy to corresponding questions, which is vital for improving "
            "Slovak language search systems."
        ),
        reference="https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad",
        dataset={
            "path": "mteb/SKQuadRetrieval",
            "revision": "8c8f19417e986ea1214310ca890f53e1c9823199",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2024-05-30", "2024-06-13"),
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
