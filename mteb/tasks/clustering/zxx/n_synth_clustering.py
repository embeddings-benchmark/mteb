from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class NSynthInstrumentFamilyClustering(AbsTaskClustering):
    label_column_name: str = "instrument_family"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="NSynthInstrumentFamilyClustering",
        description=(
            "Audio clustering of NSynth musical notes by instrument family: bass, "
            "brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. "
            "Each clip is a single four-second note, and the task evaluates whether "
            "unsupervised grouping recovers the instrument family that produced it. "
            "This is distinct from the existing NSynth classification task, which "
            "predicts instrument source (acoustic, electronic or synthetic) rather "
            "than instrument family."
        ),
        reference="https://arxiv.org/abs/1704.01279",
        dataset={
            "path": "mteb/nsynth-mini",
            "revision": "e32dfe9b65e121e64229a821fe1ff177e8962635",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2017-04-05", "2017-04-05"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        adapted_from=["NSynth"],
        bibtex_citation=r"""
@misc{engel2017neuralaudiosynthesismusical,
  archiveprefix = {arXiv},
  author = {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Douglas Eck and Karen Simonyan and Mohammad Norouzi},
  eprint = {1704.01279},
  primaryclass = {cs.LG},
  title = {Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders},
  url = {https://arxiv.org/abs/1704.01279},
  year = {2017},
}
""",
    )
