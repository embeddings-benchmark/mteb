from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class CREMADClustering(AbsTaskClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="CREMA_DClustering",
        description="Emotion clustering task with audio data for 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.",
        reference="https://ieeexplore.ieee.org/document/6849440",
        dataset={
            "path": "mteb/crema-d",
            "revision": "a9d0821955c418752248b8310438854e56a1cf34",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2014-12-31"),
        domains=["Speech"],
        task_subtypes=["Emotion Clustering"],
        license="http://opendatacommons.org/licenses/odbl/1.0/",  # Open Database License
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{Cao2014-ih,
  author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur,
Ruben C and Nenkova, Ani and Verma, Ragini},
  copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
  journal = {IEEE Transactions on Affective Computing},
  keywords = {Emotional corpora; facial expression; multi-modal recognition;
voice expression},
  language = {en},
  month = oct,
  number = {4},
  pages = {377--390},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  title = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
  volume = {5},
  year = {2014},
}
""",
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "audio"

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"], label=self.label_column_name
        )
