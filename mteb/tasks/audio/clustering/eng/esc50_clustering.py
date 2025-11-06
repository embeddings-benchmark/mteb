from mteb.abstasks.audio.abs_task_audio_clustering import AbsTaskAudioClustering
from mteb.abstasks.task_metadata import TaskMetadata


class ESC50Clustering(AbsTaskAudioClustering):
    label_column_name: str = "target"
    metadata = TaskMetadata(
        name="ESC50Clustering",
        description=(
            "The ESC-50 dataset contains 2,000 labeled environmental audio recordings "
            "evenly distributed across 50 classes (40 clips per class). "
            "These classes are organized into 5 broad categories: animal sounds, natural soundscapes & water sounds, "
            "human (non-speech) sounds, interior/domestic sounds, and exterior/urban noises. "
            "This task evaluates unsupervised clustering performance on environmental audio recordings."
        ),
        reference="https://huggingface.co/datasets/ashraq/esc50",
        dataset={
            "path": "ashraq/esc50",
            "revision": "e3e2a63ffff66b9a9735524551e3818e96af03ee",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2023-01-07", "2023-01-07"),
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Clustering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{piczak2015dataset,
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  isbn = {978-1-4503-3459-4},
  location = {{Brisbane, Australia}},
  pages = {1015--1018},
  publisher = {{ACM Press}},
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
}
""",
    )
    max_fraction_of_documents_to_embed = None
