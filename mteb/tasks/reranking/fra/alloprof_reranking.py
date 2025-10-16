from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class AlloprofReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlloprofReranking",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "mteb/AlloprofReranking",
            "revision": "a7d2d793f2e5ba55139bb10088c2e8ee2df2ce02",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map_at_1000",
        date=("2020-01-01", "2023-04-14"),  # supposition
        domains=["Web", "Academic", "Written"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{lef23,
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  doi = {10.48550/ARXIV.2302.07738},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  url = {https://arxiv.org/abs/2302.07738},
  year = {2023},
}
""",
    )
