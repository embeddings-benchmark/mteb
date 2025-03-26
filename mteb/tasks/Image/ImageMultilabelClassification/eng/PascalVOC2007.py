from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageMultilabelClassification import (
    AbsTaskImageMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


# NOTE: In the paper, this is grouped with linear probe tasks.
# See https://github.com/embeddings-benchmark/mteb/pull/2035#issuecomment-2661626309.
class VOC2007Classification(AbsTaskImageMultilabelClassification):
    metadata = TaskMetadata(
        name="VOC2007",
        description="Classifying bird images from 500 species.",
        reference="http://host.robots.ox.ac.uk/pascal/VOC/",
        dataset={
            "path": "HuggingFaceM4/pascal_voc",
            "name": "voc2007_main",
            "revision": "dbafdb9e1506c9c419c5c4672e409463cd21ba50",
            "trust_remote_code": True,
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="lrap",
        date=(
            "2005-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@Article{Everingham10,
            author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
            title = "The Pascal Visual Object Classes (VOC) Challenge",
            journal = "International Journal of Computer Vision",
            volume = "88",
            year = "2010",
            number = "2",
            month = jun,
            pages = "303--338",
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 4952},
            "avg_character_length": {"test": 431.4},
        },
    )

    # Override default column name in the subclass
    label_column_name: str = "classes"

    # To be removed when we want full results
    n_experiments: int = 5
