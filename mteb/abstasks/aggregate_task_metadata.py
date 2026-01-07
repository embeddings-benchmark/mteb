import logging
from datetime import datetime

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from mteb.types import (
    ISOLanguageScript,
    Languages,
    Licenses,
    Modalities,
    StrDate,
)

from .abstask import AbsTask
from .task_metadata import (
    AnnotatorType,
    MetadataDatasetDict,
    SampleCreationMethod,
    TaskDomain,
    TaskMetadata,
    TaskSubtype,
    TaskType,
)

logger = logging.getLogger(__name__)


class AggregateTaskMetadata(TaskMetadata):
    """Metadata for an aggregation of tasks.

    This description only covers exceptions to the TaskMetadata. Many of the field if not filled out will be autofilled from its tasks.

    Attributes:
        name: The name of the aggregated task.
        description: A description of the task. Should explain the aggregation.
        prompt: An aggregate task does not have a prompt, thus this value is always None.
        dataset: The dataset for the aggregated task is specified in its tasks.
            The aggregate task thus only specified the revision and uses a placeholder path.
        tasks: A list of tasks, the majority of the metadata is described within its tasks.
        eval_splits: The splits of the tasks used for evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    dataset: MetadataDatasetDict = MetadataDatasetDict(
        path="aggregate tasks do not have a path",  # just a place holder
        revision="1",
    )

    tasks: list[AbsTask]
    main_score: str
    type: TaskType
    eval_splits: list[str]
    eval_langs: Languages = Field(default_factory=list)
    prompt: None = None
    reference: str | None = None
    bibtex_citation: str | None = None

    @model_validator(mode="after")
    def _compute_unfilled_cases(self) -> Self:
        if not self.eval_langs:
            self.eval_langs = self._compute_eval_langs()
        if not self.date:
            self.date = self._compute_date()
        if not self.domains:
            self.domains = self._compute_domains()
        if not self.task_subtypes:
            self.task_subtypes = self._compute_task_subtypes()
        if not self.license:
            self.license = self._compute_license()
        if not self.annotations_creators:
            self.annotations_creators = self._compute_annotations_creators()
        if not self.dialect:
            self.dialect = self._compute_dialect()
        if not self.sample_creation:
            self.sample_creation = self._compute_sample_creation()
        if not self.modalities:
            self.modalities = self._compute_modalities()

        return self

    def _compute_eval_langs(self) -> list[ISOLanguageScript]:
        langs = set()
        for task in self.tasks:
            langs.update(set(task.metadata.bcp47_codes))
        return list(langs)

    def _compute_date(self) -> tuple[StrDate, StrDate] | None:
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

    def _compute_domains(self) -> list[TaskDomain] | None:
        domains = set()
        for task in self.tasks:
            if task.metadata.domains:
                domains.update(set(task.metadata.domains))
        if domains:
            return list(domains)
        return None

    def _compute_task_subtypes(self) -> list[TaskSubtype] | None:
        subtypes = set()
        for task in self.tasks:
            if task.metadata.task_subtypes:
                subtypes.update(set(task.metadata.task_subtypes))
        if subtypes:
            return list(subtypes)
        return None

    def _compute_license(self) -> Licenses | None:
        licenses = set()
        for task in self.tasks:
            if task.metadata.license:
                licenses.add(task.metadata.license)
        if len(licenses) > 1:
            return "multiple"
        return None

    def _compute_annotations_creators(self) -> AnnotatorType | None:
        creators = set()
        for task in self.tasks:
            if task.metadata.annotations_creators:
                creators.add(task.metadata.annotations_creators)
        if len(creators) > 1:
            logger.warning(
                f"Multiple annotations_creators found for tasks in {self.name}. Using None as annotations_creators."
            )
        return None

    def _compute_dialect(self) -> list[str] | None:
        dialects = set()
        for task in self.tasks:
            if task.metadata.dialect:
                dialects.update(set(task.metadata.dialect))
        if dialects:
            return list(dialects)
        return None

    def _compute_sample_creation(self) -> SampleCreationMethod | None:
        sample_creations = set()
        for task in self.tasks:
            if task.metadata.sample_creation:
                sample_creations.add(task.metadata.sample_creation)
        if len(sample_creations) > 1:
            return "multiple"
        return None

    def _compute_modalities(self) -> list[Modalities]:
        modalities = set()
        for task in self.tasks:
            if task.metadata.modalities:
                modalities.update(set(task.metadata.modalities))
        if modalities:
            return list(modalities)
        return []
