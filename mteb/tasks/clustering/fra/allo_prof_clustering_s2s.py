from mteb.abstasks.clustering import (
    AbsTaskClustering,
)
from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class AlloProfClusteringS2S(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="AlloProfClusteringS2S",
        description="Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "mteb/AlloProfClusteringS2S",
            "revision": "65e8374374d699feeffb65012a42e033d48ede38",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        date=("1996-01-01", "2023-04-14"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="human-annotated",
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
        superseded_by="AlloProfClusteringS2S.v2",
    )


class AlloProfClusteringS2SFast(AbsTaskClustering):
    max_depth = 1
    max_document_to_embed = 2556
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="AlloProfClusteringS2S.v2",
        description="Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "mteb/AlloProfClusteringS2S.v2",
            "revision": "c802b993cf2a8dcaeadb6f9cc8ccd364e251feaa",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        # (date of founding of the dataset source site, date of dataset paper publication)
        date=("1996-01-01", "2023-04-14"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
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
        adapted_from=["AlloProfClusteringS2S"],
    )
