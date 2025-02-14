from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class TUBerlinT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="TUBerlinT2IRetrieval",
        description="Retrieve sketch images based on text descriptions.",
        reference="https://dl.acm.org/doi/pdf/10.1145/2185520.2185540",
        dataset={
            "path": "gowitheflow/tu-berlin",
            "revision": "0cd78cd1ddbd3cafa9f319c638ebd77836ec9ff6",
            # "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2012-01-01", "2012-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{eitz2012humans,
  title={How do humans sketch objects?},
  author={Eitz, Mathias and Hays, James and Alexa, Marc},
  journal={ACM Transactions on graphics (TOG)},
  volume={31},
  number={4},
  pages={1--10},
  year={2012},
  publisher={Acm New York, NY, USA}
}""",
    )
