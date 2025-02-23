from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EDIST2ITRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="EDIST2ITRetrieval",
        description="Retrieve news images and titles based on news content.",
        reference="https://aclanthology.org/2023.emnlp-main.297/",
        dataset={
            "path": "MRBench/mbeir_edis_task2",
            "revision": "68c47ef3e49ef883073b3358bd4243eeca0aee9a",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["News"],
        task_subtypes=["image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{liu2023edis,
  title={EDIS: Entity-Driven image Search over Multimodal Web Content},
  author={Liu, Siqi and Feng, Weixi and Fu, Tsu-Jui and Chen, Wenhu and Wang, William},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={4877--4894},
  year={2023}
}""",
        prompt={"query": "Identify the news photo for the given caption."},
    )
