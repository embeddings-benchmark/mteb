from typing import Any

from mteb.abstasks.dialog_state_tracking import AbsTaskDST
from mteb.abstasks.task_metadata import TaskMetadata


class XRisaWozAttraction(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        # "inform-Attraction-score",
        "inform-Attraction-consumption",
        "inform-Attraction-area",
        "inform-Attraction-metro station",
        "inform-Attraction-name",
        # "inform-Attraction-opening hours",
        "inform-Attraction-type",
        # "inform-Attraction-ticket price",
    )
    subset_name = "attraction"

    metadata = TaskMetadata(
        name="XRisaWozAttraction",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozCar(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        "inform-Car-hybrid",
        "inform-Car-classification",
        "inform-Car-brand",
        # 'inform-Car-size',
        # 'inform-Car-name',
        # 'inform-Car-number of seats',
        "inform-Car-pricerange",
        "inform-Car-series",
        # 'inform-Car-4WD',
        # 'inform-Car-power level',
    )
    subset_name = "car"

    metadata = TaskMetadata(
        name="XRisaWozCar",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozClass(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        # 'inform-Class-campus',
        "inform-Class-level",
        "inform-Class-subject",
        "inform-Class-area",
        "inform-Class-day",
        "inform-Class-grade",
        # 'inform-Class-hours',
        "inform-Class-time",
    )
    subset_name = "class"

    metadata = TaskMetadata(
        name="XRisaWozClass",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozHospital(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        # 'inform-Hospital-metro station',
        "inform-Hospital-general or specialized",
        # 'inform-Hospital-level',
        "inform-Hospital-name",
        # 'inform-Hospital-public or private',
        "inform-Hospital-key departments",
        "inform-Hospital-area",
    )
    subset_name = "hospital"

    metadata = TaskMetadata(
        name="XRisaWozHospital",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozMovie(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        "inform-Movie-type",
        "inform-TV-star",
        # 'inform-TV-episodes',
        "inform-Movie-decade",
        "inform-TV-decade",
        "inform-TV-production country or area",
        "inform-Movie-production country or area",
        "inform-TV-type",
        "inform-Movie-star",
    )
    subset_name = "movie"

    metadata = TaskMetadata(
        name="XRisaWozMovie",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozPC(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        "inform-PC-CPU",
        "inform-PC-brand",
        "inform-PC-usage",
        "inform-PC-memory capacity",
        "inform-PC-computer type",
        "inform-PC-screen size",
        # 'inform-PC-product name',
        "inform-PC-series",
        "inform-PC-pricerange",
        # 'inform-PC-CPU model',
    )
    subset_name = "pc"

    metadata = TaskMetadata(
        name="XRisaWozPC",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozTrain(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        "inform-Train-departure",
        "inform-Train-date",
        "inform-Train-classification",
        "inform-Train-destination",
        "inform-Train-seat type",
    )
    subset_name = "train"

    metadata = TaskMetadata(
        name="XRisaWozTrain",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )


class XRisaWozTransport(AbsTaskDST):
    n_experiments = 1
    classification_columns = (
        "inform-Flight-destination",
        "inform-Flight-date",
        "inform-Weather-date",
        "inform-Weather-city",
        "inform-Flight-class cabin",
        "inform-Flight-departure",
        # 'inform-Flight-arrival time',
    )
    subset_name = "transport"

    metadata = TaskMetadata(
        name="XRisaWozTransport",
        description="",
        reference=None,
        dataset={
            "path": "DeepPavlov/XRISAWOZ",
            "revision": "main",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        # eval_splits=["test", "dev"],
        eval_splits=["test"],
        eval_langs={
            f"en_{subset_name}": ["eng-Latn"],
            f"fr_{subset_name}": ["fra-Latn"],
            f"enhi_{subset_name}": ["hin-Deva", "eng-Latn"],
            f"hi_{subset_name}": ["hin-Deva"],
            f"ko_{subset_name}": ["kor-Hang"],
        },
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt=None,
    )

    def dataset_transform(self) -> None:
        def process_history(row: dict[str, Any]) -> dict[str, Any]:
            history = row["history"]
            text = ""
            if len(history) > 0:
                for entry in history:
                    if entry["role"] == "user":
                        text += f"User: {entry['content']}\n"
                    else:
                        text += f"Assistant: {entry['content']}\n"
            text += f"User: {row['text']}"
            row["text"] = text
            row["history"] = None
            return row

        for subset in self.dataset:
            self.dataset[subset] = self.dataset[subset].map(
                process_history,
                remove_columns=["history"],
            )
