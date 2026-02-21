"""Classes for `eval.yaml`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from typing_extensions import Self

    from mteb import TaskMetadata


# limited version of eval.yaml model based on https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/eval.yaml
class EvalTaskConfig(BaseModel):
    id: str
    config: str | None = Field(
        None,
        description="HF dataset config name (typically languages), e.g. `eng`, `fra`. None for mean all configs are used. Config is also sometimes referred to as subset or in mteb as `hf_subset`.",
    )
    split: str | None = Field(
        None,
        description="HF dataset split name, e.g. `test` or `train`. None for mean all splits are used.",
    )
    # TODO how to specify retrieval with 3 configs (qrels, query, corpus)?
    field_spec: dict[str, str] | None = None


class EvalMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # TODO:
    #  prompts? (what to do with default?)
    #  default metric
    #  more metadata about task? (domains etc.)
    name: str
    description: str
    evaluation_framework: Literal["mteb"] = "mteb"
    tasks: list[EvalTaskConfig]

    @classmethod
    def create_from_task_metadata(
        cls, task_metadata: TaskMetadata, field_spec: dict[str, str]
    ) -> Self:
        eval_task_config = [
            # scores across all subsets and splits
            EvalTaskConfig(
                id=task_metadata.name,
                field_spec=field_spec,
            )
        ]
        for subset in task_metadata.hf_subsets:
            for split in task_metadata.eval_splits:
                eval_task_config.append(
                    EvalTaskConfig(
                        id=f"{task_metadata.name}_{subset}_{split}",
                        config=subset,
                        split=split,
                        field_spec=field_spec,
                    )
                )
        return cls(
            name=task_metadata.name,
            description=task_metadata.description,
            tasks=eval_task_config,
        )
