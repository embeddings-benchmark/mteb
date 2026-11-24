from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WebLINXCandidatesReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "mteb/WebLINXCandidatesReranking",
            "revision": "107fdc2402d2c4bfb2a720dfcfe1f6ff9d21151b",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["eng-Latn"],
        main_score="mrr_at_10",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{lù2024weblinx,
  archiveprefix = {arXiv},
  author = {Xing Han Lu and Zden{\v{e}}k Kasner and Siva Reddy},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  eprint = {2402.05930},
  pages = {33007--33056},
  publisher = {PMLR},
  series = {Proceedings of Machine Learning Research},
  title = {Web{LINX}: Real-World Website Navigation with Multi-Turn Dialogue},
  url = {https://proceedings.mlr.press/v235/lu24e.html},
  volume = {235},
  year = {2024},
}
""",
    )
