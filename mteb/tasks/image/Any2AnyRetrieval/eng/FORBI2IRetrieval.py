from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FORBI2I(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="FORBI2IRetrieval",
        description="Retrieve flat object images from 8 classes.",
        reference="https://github.com/pxiangwu/FORB",
        dataset={
            "path": "isaacchung/forb_retrieval",
            "revision": "26ab4bd972854becada339afc80f5f3ffc047e2b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2022-01-01", "2023-01-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@misc{wu2023forbflatobjectretrieval,
            title={FORB: A Flat Object Retrieval Benchmark for Universal image Embedding}, 
            author={Pengxiang Wu and Siman Wang and Kevin Dela Rosa and Derek Hao Hu},
            year={2023},
            eprint={2309.16249},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2309.16249}, 
        }
        """,
    )
