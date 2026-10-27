from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NSynth(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NSynth",
        description="Instrument Source Classification: one of acoustic, electronic, or synthetic.",
        reference="https://huggingface.co/datasets/anime-sh/NSYNTH_PITCH_HEAR",
        dataset={
            "path": "mteb/nsynth-mini",
            "revision": "e32dfe9b65e121e64229a821fe1ff177e8962635",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2025-03-06", "2025-03-06"),
        domains=["Music"],
        task_subtypes=["Instrument Source Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{engel2017neuralaudiosynthesismusical,
  archiveprefix = {arXiv},
  author = {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Mohammad Norouzi and Douglas Eck and Karen Simonyan},
  booktitle = {ICML},
  eprint = {1704.01279},
  pages = {1068-1077},
  title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
  url = {http://proceedings.mlr.press/v70/engel17a.html},
  year = {2017},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "instrument_source"
