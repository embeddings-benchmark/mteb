"""Based on https://huggingface.co/docs/hub/eval-results#adding-evaluation-results"""

import datetime
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel


class HFEvalResultDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Repo name")
    task_id: str = Field(..., description="Task name that defined in eval.yaml ids")
    revision: str | None = Field(
        ..., description="Revision of the dataset used for evaluation"
    )


class HFEvalResultSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    name: str | None = Field(
        None,
        description="Display name of results source that will be displayed on benchmark on source hover",
    )
    user: str | None = Field(None, description="HF username or organization name")


class HFEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: HFEvalResultDataset
    value: float
    verify_token: str | None = None
    date: datetime.datetime | None = None
    notes: str | None = Field(
        "Created by MTEB",
        description="Any notes about the evaluation, e.g. about the model or evaluation process. Will be displayed on benchmark on score on hover.",
    )
    source: HFEvalResultSource | None = None

    def to_dict(self) -> dict[str, Any]:
        result = self.model_dump(exclude_none=True)
        if self.date:
            result["date"] = self.date.date().isoformat()
        return result


class HFEvalResults(RootModel[list[HFEvalResult]]):
    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, item: int) -> HFEvalResult:
        return self.root[item]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            [result.to_dict() for result in self.root],
            sort_keys=False,
        )
