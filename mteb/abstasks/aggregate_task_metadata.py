from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import ConfigDict, model_validator

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.TaskMetadata import (
    ANNOTATOR_TYPE,
    LANGUAGES,
    MODALITIES,
    SAMPLE_CREATION_METHOD,
    TASK_DOMAIN,
    TASK_SUBTYPE,
    TASK_TYPE,
    HFSubset,
    TaskMetadata,
)
from mteb.custom_validators import LICENSES, STR_DATE
from mteb.languages import ISO_LANGUAGE_SCRIPT

logger = logging.getLogger(__name__)


class AggregateTaskMetadata(TaskMetadata):
    """Metadata for an aggregation of tasks. This description only covers exceptions to the TaskMetadata. Many of the field if not filled out will be
    autofilled from its tasks.

    Attributes:
        name: The name of the aggregated task.
        description: A description of the task. Should explain the aggregation.
        prompt: An aggregate task does not have a prompt, thus this value is always None.
        dataset: The dataset for the aggregated task is specified in its tasks. The aggregate task thus only specified the revision and uses a
            placeholder path.
        tasks: A list of tasks, the majority of the metadata is described within its tasks.
        eval_splits: The splits of the tasks used for evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    dataset: dict[str, Any] = {
        "path": "aggregate tasks do not have a path",  # just a place holder
        "revision": "1",
    }

    tasks: list[AbsTask]
    main_score: str
    type: TASK_TYPE
    eval_splits: list[str]
    eval_langs: LANGUAGES = []
    prompt: None = None
    reference: str | None = None
    bibtex_citation: str | None = None

    @property
    def hf_subsets_to_langscripts(self) -> dict[HFSubset, list[ISO_LANGUAGE_SCRIPT]]:
        """Return a dictionary mapping huggingface subsets to languages."""
        if isinstance(self.eval_langs, dict):
            langs = []
            for v in self.eval_langs.values():
                langs.extend(v)
            langs = list(set(langs))
            return {"default": langs}
        return {"default": self.eval_langs}  # type: ignore

    @model_validator(mode="after")  # type: ignore
    def compute_unfilled_cases(self) -> AggregateTaskMetadata:
        if not self.eval_langs:
            self.eval_langs = self.compute_eval_langs()
        if not self.date:
            self.date = self.compute_date()
        if not self.domains:
            self.domains = self.compute_domains()
        if not self.task_subtypes:
            self.task_subtypes = self.compute_task_subtypes()
        if not self.license:
            self.license = self.compute_license()
        if not self.annotations_creators:
            self.annotations_creators = self.compute_annotations_creators()
        if not self.dialect:
            self.dialect = self.compute_dialect()
        if not self.sample_creation:
            self.sample_creation = self.compute_sample_creation()
        if not self.modalities:
            self.modalities = self.compute_modalities()

        return self

    def compute_eval_langs(self) -> list[ISO_LANGUAGE_SCRIPT]:
        langs = set()
        for task in self.tasks:
            langs.update(set(task.metadata.bcp47_codes))
        return list(langs)

    def compute_date(self) -> tuple[STR_DATE, STR_DATE] | None:
        # get min max date from tasks
        dates = []
        for task in self.tasks:
            if task.metadata.date:
                dates.append(datetime.fromisoformat(task.metadata.date[0]))
                dates.append(datetime.fromisoformat(task.metadata.date[1]))

        if not dates:
            return None

        min_date = min(dates)
        max_date = max(dates)
        return min_date.isoformat(), max_date.isoformat()

    def compute_domains(self) -> list[TASK_DOMAIN] | None:
        domains = set()
        for task in self.tasks:
            if task.metadata.domains:
                domains.update(set(task.metadata.domains))
        if domains:
            return list(domains)
        return None

    def compute_task_subtypes(self) -> list[TASK_SUBTYPE] | None:
        subtypes = set()
        for task in self.tasks:
            if task.metadata.task_subtypes:
                subtypes.update(set(task.metadata.task_subtypes))
        if subtypes:
            return list(subtypes)
        return None

    def compute_license(self) -> LICENSES | None:
        licenses = set()
        for task in self.tasks:
            if task.metadata.license:
                licenses.add(task.metadata.license)
        if len(licenses) > 1:
            return "multiple"
        return None

    def compute_annotations_creators(self) -> ANNOTATOR_TYPE | None:
        creators = set()
        for task in self.tasks:
            if task.metadata.annotations_creators:
                creators.add(task.metadata.annotations_creators)
        if len(creators) > 1:
            logger.warning(
                f"Multiple annotations_creators found for tasks in {self.name}. Using None as annotations_creators."
            )
        return None

    def compute_dialect(self) -> list[str] | None:
        dialects = set()
        for task in self.tasks:
            if task.metadata.dialect:
                dialects.update(set(task.metadata.dialect))
        if dialects:
            return list(dialects)
        return None

    def compute_sample_creation(self) -> SAMPLE_CREATION_METHOD | None:
        sample_creations = set()
        for task in self.tasks:
            if task.metadata.sample_creation:
                sample_creations.add(task.metadata.sample_creation)
        if len(sample_creations) > 1:
            return "multiple"
        return None

    def compute_modalities(self) -> list[MODALITIES]:
        modalities = set()
        for task in self.tasks:
            if task.metadata.modalities:
                modalities.update(set(task.metadata.modalities))
        if modalities:
            return list(modalities)
        return None
