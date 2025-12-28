import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mteb._evaluators import BitextMiningEvaluator
from mteb.abstasks._statistics_calculation import calculate_text_statistics
from mteb.abstasks.abstask import AbsTask
from mteb.models import EncoderProtocol, MTEBModels
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import SplitDescriptiveStatistics, TextStatistics

logger = logging.getLogger(__name__)


class BitextDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Bitext

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of duplicate pairs

        sentence1_statistics: Statistics for sentence1
        sentence2_statistics: Statistics for sentence2
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    sentence1_statistics: TextStatistics
    sentence2_statistics: TextStatistics


class BitextMiningMetrics(TypedDict):
    """Metrics for BitextMining tasks

    Attributes:
        precision: Precision of the model.
        recall: Recall of the model.
        f1: F1 score of the model.
        accuracy: Accuracy of the model.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float


class AbsTaskBitextMining(AbsTask):
    """Abstract class for BitextMining tasks

    The similarity is computed between pairs and the results are ranked.

    Attributes:
        dataset: A HuggingFace dataset containing the data for the task. It must contain the following columns sentence1 and sentence2 for the two texts to be compared.
        parallel_subsets: If true task language pairs should be in one split as column names, otherwise each language pair should be a subset.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    parallel_subsets = False
    abstask_prompt = "Retrieve parallel sentences."
    _DEFAULT_PAIR: ClassVar[list[tuple[str, str]]] = [("sentence1", "sentence2")]

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Added load for "parallel" datasets"""
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        if not self.data_loaded:
            self.load_data()

        hf_subsets = self.hf_subsets

        # If subsets_to_run is specified, filter the hf_subsets accordingly
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        encoder_model = cast(EncoderProtocol, model)

        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")

        scores: dict[str, BitextMiningMetrics] = {}
        if self.parallel_subsets:
            scores = self._evaluate_subset(  # type: ignore[assignment]
                encoder_model,
                self.dataset[split],
                parallel=True,
                hf_split=split,
                hf_subset="parallel",
                encode_kwargs=encode_kwargs,
                prediction_folder=prediction_folder,
                **kwargs,
            )
        else:
            for hf_subset in hf_subsets:
                logger.info(
                    f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
                )

                if hf_subset not in self.dataset and hf_subset == "default":
                    data_split = self.dataset[split]
                else:
                    data_split = self.dataset[hf_subset][split]
                scores[hf_subset] = self._evaluate_subset(  # type: ignore[assignment]
                    encoder_model,
                    data_split,
                    hf_split=split,
                    hf_subset=hf_subset,
                    encode_kwargs=encode_kwargs,
                    prediction_folder=prediction_folder,
                    **kwargs,
                )

        return cast(dict[HFSubset, ScoresDict], scores)

    def _get_pairs(self, parallel: bool) -> list[tuple[str, str]]:
        pairs = self._DEFAULT_PAIR
        if parallel:
            pairs = [langpair.split("-") for langpair in self.hf_subsets]  # type: ignore[misc]
        return pairs

    def _evaluate_subset(  # type: ignore[override]
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        parallel: bool = False,
        **kwargs,
    ) -> BitextMiningMetrics | dict[str, BitextMiningMetrics]:
        pairs = self._get_pairs(parallel)

        evaluator = BitextMiningEvaluator(
            data_split,
            task_metadata=self.metadata,
            pair_columns=pairs,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        # NOTE: used only by BUCC
        gold = (
            list(zip(range(len(data_split)), range(len(data_split))))
            if "gold" not in data_split
            else data_split["gold"]
        )

        neighbours = evaluator(model, encode_kwargs=encode_kwargs)

        if prediction_folder:
            self._save_task_predictions(
                neighbours,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        if parallel:
            parallel_metrics = {}
            for keys, nearest_neighbors in neighbours.items():
                parallel_metrics[keys] = self._compute_metrics(nearest_neighbors, gold)

            for v in parallel_metrics.values():
                self._add_main_score(v)
            return parallel_metrics
        def_pair_str = "-".join(self._DEFAULT_PAIR[0])
        metrics = self._compute_metrics(neighbours[def_pair_str], gold)
        self._add_main_score(metrics)
        return metrics

    def _compute_metrics(
        self,
        nearest_neighbors: list[dict[str, float]],
        gold: list[tuple[int, int]],
    ) -> BitextMiningMetrics:
        logger.info("Computing metrics...")
        labels = []
        predictions = []
        for i, x in enumerate(nearest_neighbors):
            j = x["corpus_id"]
            predictions.append(j)
            labels.append(gold[i][1])

        return BitextMiningMetrics(
            precision=precision_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            recall=recall_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            f1=f1_score(labels, predictions, zero_division=0, average="weighted"),
            accuracy=accuracy_score(labels, predictions),
        )

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> BitextDescriptiveStatistics:
        pairs_cols = self._get_pairs(self.parallel_subsets)
        if hf_subset:
            if self.parallel_subsets:
                sent_1, sent_2 = hf_subset.split("-")
                sentence1 = self.dataset[split][sent_1]
                sentence2 = self.dataset[split][sent_2]
            else:
                sent_1, sent_2 = pairs_cols[0]
                sentence1 = self.dataset[hf_subset][split][sent_1]
                sentence2 = self.dataset[hf_subset][split][sent_2]
        elif compute_overall:
            sentence1, sentence2 = [], []
            if self.parallel_subsets:
                for hf_subset in self.metadata.eval_langs:
                    sent_1, sent_2 = hf_subset.split("-")
                    sentence1.extend(self.dataset[split][sent_1])
                    sentence2.extend(self.dataset[split][sent_2])
            else:
                sent_1, sent_2 = pairs_cols[0]
                for hf_subset in self.metadata.eval_langs:
                    sentence1.extend(self.dataset[hf_subset][split][sent_1])
                    sentence2.extend(self.dataset[hf_subset][split][sent_2])
        else:
            sent_1, sent_2 = pairs_cols[0]
            sentence1 = self.dataset[split][sent_1]
            sentence2 = self.dataset[split][sent_2]

        text1_statistics = calculate_text_statistics(sentence1)
        text2_statistics = calculate_text_statistics(sentence2)
        unique_pairs = len(set(zip(sentence1, sentence2)))

        return BitextDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=(
                text1_statistics["total_text_length"]
                + text2_statistics["total_text_length"]
            ),
            unique_pairs=unique_pairs,
            sentence1_statistics=text1_statistics,
            sentence2_statistics=text2_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")

        if self.metadata.is_multilingual:
            dataset: dict[str, dict[str, list[str]]] = defaultdict(dict)
            for config in self.metadata.eval_langs:
                logger.info(f"Converting {config} of {self.metadata.name}")

                if self.parallel_subsets:
                    for split in self.dataset:
                        sent_1, sent_2 = config.split("-")
                        dataset[split][sent_1] = self.dataset[split][sent_1]
                        dataset[split][sent_2] = self.dataset[split][sent_2]
                else:
                    sent_1, sent_2 = self._get_pairs(self.parallel_subsets)[0]
                    lang_1, lang_2 = config.split("-")
                    for split in self.dataset[config]:
                        dataset[split][lang_1] = self.dataset[config][split][sent_1]
                        dataset[split][lang_2] = self.dataset[config][split][sent_2]
            dataset_dict = DatasetDict(
                {split: Dataset.from_dict(dataset[split]) for split in dataset}
            )
            dataset_dict.push_to_hub(repo_name)
        else:
            sentences = {}
            for split in self.dataset:
                sent_1, sent_2 = self._get_pairs(self.parallel_subsets)[0]
                sentences[split] = Dataset.from_dict(
                    {
                        "sentence1": self.dataset[split][sent_1],
                        "sentence2": self.dataset[split][sent_2],
                    }
                )
            sentences = DatasetDict(sentences)
            sentences.push_to_hub(repo_name)
