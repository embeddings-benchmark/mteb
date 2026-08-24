from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import patch

from mteb._create_dataloaders import create_dataloader as _create_dataloader
from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset

    from mteb.models.models_protocols import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs


class EMIDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="EMIDPairClassification",
        description=(
            "Pair classification on EMID: determining whether a music clip and an "
            "image share the same emotion category. Positive pairs are true "
            "audio–image matches from EMID; negatives pair a clip with an image "
            "from a different emotion."
        ),
        reference="https://arxiv.org/abs/2308.07622",
        dataset={
            "path": "Wissam42/EMID-PC-AI",
            "revision": "7758c20e79039527e717939e1c7590777f989489",
        },
        type="AudioPairClassification",
        category="a2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-01-01", "2023-08-01"),
        domains=["Music", "Web"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{zhang2023emid,
  author = {Zhang, Yujie and others},
  journal = {arXiv preprint arXiv:2308.07622},
  title = {Emotionally Paired Music and Image Dataset},
  year = {2023},
}
""",
        is_beta=True,
    )

    input1_column_name: ClassVar[dict[str, str]] = {"audio": "audio"}
    input2_column_name: ClassVar[dict[str, str]] = {"image": "image"}
    label_column_name: str = "label"
    # category a2i → query prepares audio only, document prepares image only
    input1_prompt_type = PromptType.query
    input2_prompt_type = PromptType.document

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> dict[str, float]:
        # Shared PairClassificationEvaluator omits prompt_type when building
        # dataloaders, so asymmetric a2i still tries to prepare both modalities
        # and KeyErrors on the audio-only side. Inject side prompt types.
        prompts = iter([self.input1_prompt_type, self.input2_prompt_type])

        def create_dataloader(dataset: Dataset, **dl_kwargs: Any):
            dl_kwargs["prompt_type"] = next(prompts)
            return _create_dataloader(dataset, **dl_kwargs)

        with patch(
            "mteb._evaluators.pair_classification_evaluator.create_dataloader",
            create_dataloader,
        ):
            return super()._evaluate_subset(
                model,
                data_split,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                num_proc=num_proc,
                timer=timer,
                **kwargs,
            )
