"""Based on https://huggingface.co/docs/hub/eval-results#adding-evaluation-results"""

import datetime

from pydantic import BaseModel, ConfigDict, Field


class EvalResultDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Repo name")
    task_id: str = Field(..., description="Task name that defined in eval.yaml ids")
    revision: str | None


class EvalResultSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    name: str | None = Field(
        None, description="Display name"
    )  # What mean display name?
    user: str | None = Field(None, description="HF username or organization name")


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: EvalResultDataset
    metric: float
    verify_token: str | None = None
    date: datetime.date | None = Field(
        None, description="evaluation ran in HF Jobs with inspect-ai"
    )
    notes: str | None = Field(
        "Created by mteb",
        description="any notes about the evaluation, e.g. about the model or evaluation process",
    )
    source: EvalResultSource | None = None
