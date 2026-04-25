from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class HMDB51Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="HMDB51Clustering",
        description=(
            "Clustering of video clips into 51 human action categories from "
            "HMDB51, a large video database for human motion recognition."
        ),
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "73e5ac9cd9536c406d0046f3d6046785885f7ebe",
        },
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2011-01-01", "2011-12-31"),
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kuehne2011hmdb,
    title = {HMDB: a large video database for human motion recognition},
    author = {Kuehne, Hildegard and Jhuang, Hueihan and Garrote, Est{\'\i}baliz and Poggio, Tomaso and Serre, Thomas},
    booktitle = {2011 International Conference on Computer Vision},
    pages = {2556--2563},
    year = {2011},
    organization = {IEEE},
}
""",
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "label"],
            )
