"""Classes for `eval.yaml`.

Limited version of eval.yaml model based on https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/eval.yaml
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class HFEvalTaskConfig(BaseModel):
    id: str
    config: str | None = Field(
        None,
        description="HF dataset config name (typically languages), e.g. `eng`, `fra`. None for mean all configs are used. Config is also sometimes referred to as subset or in mteb as `hf_subset`.",
    )
    split: str | None = Field(
        None,
        description="HF dataset split name, e.g. `test` or `train`. None for mean all splits are used.",
    )


class HFEvalMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # TODO:
    #  prompts? (what to do with default?)
    #  default metric
    #  more metadata about task? (domains etc.)
    name: str
    description: str
    evaluation_framework: Literal["mteb"] = "mteb"
    tasks: list[HFEvalTaskConfig]

    def merge(self, other: HFEvalMeta) -> HFEvalMeta:
        existing_tasks = {t.id for t in self.tasks}
        for other_task in other.tasks:
            if other_task.id not in existing_tasks:
                self.tasks.append(other_task)
        self.tasks = sorted(self.tasks, key=lambda t: t.id)
        return self

    def to_yaml(self) -> str:
        task_config_dict = self.model_dump(exclude_none=True)
        return yaml.safe_dump(
            task_config_dict,
            # to keep order id, split, config
            sort_keys=False,
        )
