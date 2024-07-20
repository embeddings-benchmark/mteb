from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks import AbsTaskImageClassification

## mteb run -m openai/clip-vit-base-patch32 --model_revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 --output_folder results-mieb -t MNIST


class MNISTClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="MNIST",
        description="Classifying handwritten digits.",
        reference="https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html",
        dataset={
            "path": "ylecun/mnist",
            "revision": "b06aab39e05f7bcd9635d18ed25d06eae523c574",
        },
        type="Classification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2010-04-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@article{lecun2010mnist,
        title={MNIST handwritten digit database},
        author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
        journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
        volume={2},
        year={2010}
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {"test": 431.4},
        },
    )
