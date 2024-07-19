from __future__ import annotations

import logging
from typing import Any, List, Union

from datasets import Dataset
from tqdm import tqdm
from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.evaluation.evaluators import ImageTextPairClassificationEvaluator
from mteb.load_results.mteb_results import ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskImageTextPairClassification(AbsTask):
    """Abstract class for Image Text Pair Classification tasks,
    e.g. Compositionality evaluation.
    The similarity is computed between pairs and the results are ranked.
    Note that the number of images and the number of captions can be different.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        images: List[List[Image.Image]]
        captions: List[List[str]]
    """

    # it can be ["image_0", "image_1"]; ["text_0", "text_1"] for datasets like WinoGround
    images_column_names: Union[str, List[str]] = "image"
    texts_column_names: Union[str, List[str]] = "caption"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_column(self, dataset: Dataset, column_names: Union[str, List[str]]) -> List[List[Any]]:
        """Group examples from the columns into a list of examples."""
        if isinstance(column_names, str):
            return dataset[column_names]
        
        return [
            [example[col] for col in column_names] 
            for example in tqdm(dataset, desc=f"Processing columns {column_names}")
        ]
         
    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:

        images = self._preprocess_column(dataset, self.images_column_names)
        texts = self._preprocess_column(dataset, self.texts_column_names)

        evaluator = ImageTextPairClassificationEvaluator(
            images,
            texts,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores
