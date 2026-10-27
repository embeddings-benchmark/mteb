from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FORBI2I(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FORBI2IRetrieval",
        description="Retrieve flat object images from 8 classes.",
        reference="https://github.com/pxiangwu/FORB",
        dataset={
            "path": "mteb/forb_retrieval",
            "revision": "26ab4bd972854becada339afc80f5f3ffc047e2b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2022-01-01", "2023-01-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{wu2023forbflatobjectretrieval,
  archiveprefix = {arXiv},
  author = {Pengxiang Wu and Siman Wang and Kevin S Dela Rosa and Derek Hao Hu},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  eprint = {2309.16249},
  title = {{FORB}: A Flat Object Retrieval Benchmark for Universal Image Embedding},
  url = {https://openreview.net/forum?id=c3kuX7ltzr},
  year = {2023},
}
""",
    )
