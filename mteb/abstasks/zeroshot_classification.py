from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset
from typing_extensions import deprecated

from .abstask import AbsTask
from .retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.timing import TimingStack
    from mteb.types import Modalities

    from .retrieval_dataset_loaders import RetrievalSplitData


logger = logging.getLogger(__name__)


@deprecated(
    "This class is deprecated and will be removed in future versions. Please use the updated retrieval tasks instead. "
    "You can reupload it in the retrieval format using `task.push_dataset_to_hub('your/repository')` after loading the data."
)
class AbsTaskZeroShotClassification(AbsTaskRetrieval):
    """Abstract class for ZeroShot Classification tasks for any modality.

    Warning: Deprecated
        This class is deprecated and will be removed in future versions. Please use [AbsTaskRetrieval][mteb.abstasks.retrieval.AbsTaskRetrieval] instead.
        You can reupload it in the retrieval format using `task.push_dataset_to_hub('your/repository')` after loading the data.

    The similarity between an input (can be image or text) and candidate text prompts, such as this is a dog/this is a cat.

    Zero-shot classification is a retrieval task: queries are the inputs, the corpus is the set of candidate labels and
    the only relevant document of a query is its correct label. Datasets stored as input and label columns are converted
    to this format on the fly, so classification metrics (`accuracy`, `f1`, ...) are reported next to the retrieval ones.
    For the dataformat and other information, see [AbsTaskRetrieval][mteb.abstasks.retrieval.AbsTaskRetrieval].

    Attributes:
        dataset: Huggingface dataset containing the data for the task. Dataset must contain columns specified by self.input_column_name and self.label_column_name.
            After loading it is converted to the retrieval format.
        input_column_name: Name of the column containing the inputs (image or text).
        label_column_name: Name of the column containing the labels. Labels must be
            integer indices of the candidate labels or strings matching an entry of
            `get_candidate_labels`.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    input_column_name: str | Sequence[Modalities] = "image"
    label_column_name: str = "label"
    abstask_prompt = "Classify the given input into one of the candidate labels."
    # the correct label is the only relevant document, so there is nothing to rank beyond the label set
    k_values: Sequence[int] = (1, 3, 5, 10)

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the dataset and convert it to the retrieval format.

        Datasets with input and label columns are converted: inputs become queries, candidate labels become the corpus
        and the correct label of a query is its only relevant document.
        """
        if self.data_loaded:
            return

        # a task can provide the raw dataset itself (e.g. mock tasks)
        if self.dataset is None:
            AbsTask.load_data(self, num_proc=num_proc, timer=timer, **kwargs)

        candidate_labels = self.get_candidate_labels()
        corpus = Dataset.from_dict(
            {
                "id": [str(i) for i in range(len(candidate_labels))],
                "text": candidate_labels,
            }
        )
        # a multimodal task names each of its input columns after the modality it holds
        if isinstance(self.input_column_name, str):
            # the input is the non-text modality, text is only used for the candidate labels
            query_modality = next(
                (modality for modality in self.modalities if modality != "text"),
                "text",
            )
            columns = {self.input_column_name: query_modality}
        else:
            columns = {column: column for column in self.input_column_name}

        def _convert_split(split_data: Dataset) -> RetrievalSplitData:
            labels = self._normalize_labels(
                split_data[self.label_column_name], candidate_labels
            )
            queries = split_data.select_columns(list(columns)).rename_columns(columns)
            queries = queries.add_column("id", [str(i) for i in range(len(queries))])
            return {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": {
                    str(i): {str(label): 1} for i, label in enumerate(labels)
                },
                "top_ranked": None,
            }

        eval_splits = set(self.metadata.eval_splits)
        if self.metadata.is_multilingual:
            self.dataset = {
                subset: {
                    split: _convert_split(data)
                    for split, data in subset_data.items()
                    if split in eval_splits
                }
                for subset, subset_data in self.dataset.items()
            }
        else:
            self.dataset = {
                "default": {
                    split: _convert_split(data)
                    for split, data in self.dataset.items()
                    if split in eval_splits
                }
            }
        self.data_loaded = True

    @staticmethod
    def _normalize_labels(
        labels: list[int] | list[str], candidate_labels: list[str]
    ) -> list[int]:
        """Convert dataset labels to integer indices of the candidate labels.

        Predictions are always integer indices into ``candidate_labels``, while
        datasets store labels either as integer class indices (e.g. a
        ``ClassLabel`` column) or as strings. scikit-learn >= 1.9 raises an
        error when ``y_true`` contains strings and ``y_pred`` is numeric, so
        string labels are mapped to their index in ``candidate_labels``.

        Args:
            labels: Labels as stored in the dataset.
            candidate_labels: Candidate labels returned by `get_candidate_labels`.

        Returns:
            Labels as integer indices into ``candidate_labels``.

        Raises:
            ValueError: If a string label does not match any candidate label.
        """
        if not labels or not isinstance(labels[0], str):
            return cast("list[int]", labels)

        label_to_index = {label: idx for idx, label in enumerate(candidate_labels)}
        unknown_labels = sorted(set(labels) - label_to_index.keys())
        if unknown_labels:
            raise ValueError(
                "String labels must match an entry of `get_candidate_labels` to "
                f"be mapped to a candidate index, but {unknown_labels} do not. "
                "Alternatively, store labels as integer indices of the candidate "
                "labels."
            )
        return [label_to_index[label] for label in labels]

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification.

        Only needed for datasets stored as inputs and labels. Datasets in the retrieval format keep them in the corpus.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
