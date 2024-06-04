from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast, convert_to_fast
from mteb.abstasks.TaskMetadata import TaskMetadata

NUM_SAMPLES = 2048


class BlurbsClusteringP2P(AbsTaskClustering):
    superseeded_by = "BlurbsClusteringP2P.v2"

    metadata = TaskMetadata(
        name="BlurbsClusteringP2P",
        description="Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        dataset={
            "path": "slvnwhrl/blurbs-clustering-p2p",
            "revision": "a2dd5b02a77de3466a3eaa98ae586b5610314496",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
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
        bibtex_citation=None,
        n_samples={"test": 174637},
        avg_character_length={"test": 664.09},
    )


class BlurbsClusteringP2PFast(AbsTaskClusteringFast):
    # a faster version of BlurbsClusteringP2P, since it does not sample from the same distribution we can't use the AbsTaskClusteringFast, instead we
    # simply downsample each cluster.

    metadata = TaskMetadata(
        name="BlurbsClusteringP2P.v2",
        description="Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        dataset={
            "path": "slvnwhrl/blurbs-clustering-p2p",
            "revision": "a2dd5b02a77de3466a3eaa98ae586b5610314496",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=(
            "1900-01-01",
            "2019-12-31",
        ),  # since it is books it is likely to be from the 20th century -> paper from 2019
        form=["written"],
        domains=["Fiction"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-nc-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{Remus2019GermEval2T,
  title={GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
  author={Steffen Remus and Rami Aly and Chris Biemann},
  booktitle={Conference on Natural Language Processing},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:208334484}
}""",
        n_samples={"test": NUM_SAMPLES},
        avg_character_length={"test": 664.09},
    )

    def dataset_transform(self):
        self.dataset = convert_to_fast(self.dataset, self.seed)  # type: ignore
