from mteb.abstasks import AbsTaskReranking
from mteb.abstasks.task_metadata import TaskMetadata


class RuWebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="RuWebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "DeepPavlov/WebLINX-ru",
            "revision": "fdc93113fd2d43ee6aae0fd59a53ebd7bc201287",
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
        eval_langs=["rus-Cyrl"],
        main_score="mrr_at_10",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""
@misc{lù2024weblinx,
  archiveprefix = {arXiv},
  author = {Xing Han Lù and Zdeněk Kasner and Siva Reddy},
  eprint = {2402.05930},
  primaryclass = {cs.CL},
  title = {WebLINX: Real-World Website Navigation with Multi-Turn Dialogue},
  year = {2024},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"query": "query_en", "query_ru": "query"}
        )
