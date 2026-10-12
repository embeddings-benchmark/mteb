from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Kinetics400ZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics400ZeroShot",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long. Sampled 25 random shard files from each official path list, capped at 10 per class across 400 classes (~3,970 train / ~4,000 test).",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/Kinetics400ZeroShot",
            "revision": "b6f11180b28e64339cdafc400523d0c25022acc1",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-05-19",
            "2017-05-19",
        ),
        domains=["Activity", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kay2017kineticshumanactionvideo,
  archiveprefix = {arXiv},
  author = {Will Kay and Joao Carreira and Karen Simonyan and Brian Zhang and Chloe Hillier and Sudheendra Vijayanarasimhan and Fabio Viola and Tim Green and Trevor Back and Paul Natsev and Mustafa Suleyman and Andrew Zisserman},
  eprint = {1705.06950},
  primaryclass = {cs.CV},
  title = {The Kinetics Human Action Video Dataset},
  url = {https://arxiv.org/abs/1705.06950},
  year = {2017},
}
""",
        is_beta=True,
    )


class Kinetics400VAZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Kinetics400VAZeroShot",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long. This variant uses both video and audio modalities. Sampled 25 random shard files from each official path list, capped at 10 per class across 400 classes (~3,970 train / ~4,000 test).",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/Kinetics400VAZeroShot",
            "revision": "de9024bb4ffb7334aca534c108bffdfab41cd871",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-05-19",
            "2017-05-19",
        ),
        domains=["Activity", "Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kay2017kineticshumanactionvideo,
  archiveprefix = {arXiv},
  author = {Will Kay and Joao Carreira and Karen Simonyan and Brian Zhang and Chloe Hillier and Sudheendra Vijayanarasimhan and Fabio Viola and Tim Green and Trevor Back and Paul Natsev and Mustafa Suleyman and Andrew Zisserman},
  eprint = {1705.06950},
  primaryclass = {cs.CV},
  title = {The Kinetics Human Action Video Dataset},
  url = {https://arxiv.org/abs/1705.06950},
  year = {2017},
}
""",
        is_beta=True,
    )
