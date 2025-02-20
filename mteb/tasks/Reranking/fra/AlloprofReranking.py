from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="s2p",
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
        bibtex_citation="""@misc{lef23,
            doi = {10.48550/ARXIV.2302.07738},
            url = {https://arxiv.org/abs/2302.07738},
            author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
            keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
            title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
            publisher = {arXiv},
            year = {2023},
            copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
            }""",
    )
