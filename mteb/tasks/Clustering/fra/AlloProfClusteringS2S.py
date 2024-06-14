import datasets
import numpy as np

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AlloProfClusteringS2S(AbsTaskClustering):
    superseeded_by = "AlloProfClusteringS2S.v2"

    metadata = TaskMetadata(
        name="AlloProfClusteringS2S",
        description="Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
            "name": "documents",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
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
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self):
        """Convert to standard format"""
        self.dataset = self.dataset.remove_columns("uuid")
        self.dataset = self.dataset.remove_columns("text")
        titles = self.dataset["documents"]["title"]
        topics = self.dataset["documents"]["topic"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(titles, 10)],
            "labels": [split.tolist() for split in np.array_split(topics, 10)],
        }
        self.dataset["test"] = datasets.Dataset.from_dict(new_format)
        self.dataset.pop("documents")


class AlloProfClusteringS2SFast(AbsTaskClusteringFast):
    max_depth = 1

    metadata = TaskMetadata(
        name="AlloProfClusteringS2S.v2",
        description="Clustering of document titles from Allo Prof dataset. Clustering of 10 sets on the document topic.",
        reference="https://huggingface.co/datasets/lyon-nlp/alloprof",
        dataset={
            "path": "lyon-nlp/alloprof",
            "revision": "392ba3f5bcc8c51f578786c1fc3dae648662cb9b",
            "name": "documents",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="v_measure",
        # (date of founding of the dataset source site, date of dataset paper publication)
        date=("1996-01-01", "2023-04-14"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{lef23,
  doi = {10.48550/ARXIV.2302.07738},
  url = {https://arxiv.org/abs/2302.07738},
  author = {Lefebvre-Brossard, Antoine and Gazaille, Stephane and Desmarais, Michel C.},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Alloprof: a new French question-answer education dataset and its use in an information retrieval case study},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
""",
        n_samples={"test": 2556},
        avg_character_length={"test": 32.8},
    )

    def dataset_transform(self):
        self.dataset["test"] = (
            self.dataset["documents"]
            .rename_columns({"title": "sentences", "topic": "labels"})
            .select_columns(["sentences", "labels"])
        )
        self.dataset.pop("documents")
        unique_labels = list(set(self.dataset["test"]["labels"]))
        unique_labels.sort()
        self.dataset["test"] = self.dataset["test"].cast(
            datasets.Features(
                sentences=datasets.Value("string"),
                labels=datasets.ClassLabel(names=unique_labels),
            )
        )
        for split in self.metadata.eval_splits:
            check_label_distribution(self.dataset[split])
