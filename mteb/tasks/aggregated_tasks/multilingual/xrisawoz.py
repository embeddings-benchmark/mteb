from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.dialog_state_tracking import XRisaWozPC, XRisaWozAttraction, XRisaWozCar, XRisaWozTransport, XRisaWozClass, XRisaWozMovie, XRisaWozTrain, XRisaWozHospital


class XRisaWoz(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="XRiSAWOZ",
        description="",
        reference="",
        tasks=[
            XRisaWozPC(),
            XRisaWozAttraction(),
            XRisaWozCar(),
            XRisaWozTransport(),
            XRisaWozClass(),
            XRisaWozMovie(),
            XRisaWozTrain(),
            XRisaWozHospital(),
        ],
        category="t2c",
        license="not specified",
        modalities=["text"],
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        main_score="f1",
        type="Classification",
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "fr": ["fra-Latn"],
            "enhi": ["hin-Deva", "eng-Latn"],
            "hi": ["hin-Deva"],
            "ko": ["kor-Hang"],
        },
        bibtex_citation=None,
    )
