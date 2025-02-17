from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AROCocoOrder(AbsTaskImageTextPairClassification):
    images_column_names = ["images"]
    texts_column_names = [
        "correct_caption",
        "hard_text_1",
        "hard_text_2",
        "hard_text_3",
        "hard_text_4",
    ]

    metadata = TaskMetadata(
        name="AROCocoOrder",
        description="Compositionality Evaluation of images to their captions."
        + "Each capation has four hard negatives created by order permutations.",
        reference="https://proceedings.neurips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html",
        dataset={
            "path": "gowitheflow/ARO-COCO-order",
            "revision": "853ec8757226585a38a80886c51fe0f3f268787c",
        },
        type="Compositionality",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="text_acc",
        date=(
            "2022-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of data
        domains=["Encyclopaedic"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@article{hsieh2024sugarcrepe,
  title={Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 25010},
            "avg_character_length": {"test": 1},
        },
    )
