from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CREMADPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CREMADPairClassification",
        description="Classifying pairs as having same or different emotions in actor's voice recordings of text spoken in 6 different emotions",
        reference="https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/",
        dataset={
            "path": "mteb/CREMADPairClassification",
            "revision": "20701cbd51bf282719674c9f38ec91e698b62e48",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-01-01", "2014-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="http://opendatacommons.org/licenses/odbl/1.0/",
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

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"
