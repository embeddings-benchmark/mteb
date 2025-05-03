from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast, convert_to_fast
from mteb.abstasks.TaskMetadata import TaskMetadata

NUM_SAMPLES = 2048


class BlurbsClusteringP2P(AbsTaskClustering):
    superseded_by = "BlurbsClusteringP2P.v2"

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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=["Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{Remus2019GermEval2T,
  author = {Steffen Remus and Rami Aly and Chris Biemann},
  booktitle = {Conference on Natural Language Processing},
  title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
  url = {https://api.semanticscholar.org/CorpusID:208334484},
  year = {2019},
}
""",
    )


class BlurbsClusteringP2PFast(AbsTaskClusteringFast):
    # a faster version of BlurbsClusteringP2P, since it does not sample from the same distribution we can't use the AbsTaskClusteringFast, instead we
    # simply downsample each cluster.
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="v_measure",
        date=(
            "1900-01-01",
            "2019-12-31",
        ),  # since it is books it is likely to be from the 20th century -> paper from 2019
        domains=["Fiction", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Remus2019GermEval2T,
  author = {Steffen Remus and Rami Aly and Chris Biemann},
  booktitle = {Conference on Natural Language Processing},
  title = {GermEval 2019 Task 1: Hierarchical Classification of Blurbs},
  url = {https://api.semanticscholar.org/CorpusID:208334484},
  year = {2019},
}
""",
        adapted_from=["BlurbsClusteringP2P"],
    )

    def dataset_transform(self):
        self.dataset = convert_to_fast(self.dataset, self.seed)  # type: ignore
