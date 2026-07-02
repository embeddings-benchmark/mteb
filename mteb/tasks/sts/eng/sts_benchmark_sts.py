from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_STS_BENCHMARK_DATASET = {
    "path": "mteb/stsbenchmark-sts",
    "revision": "b0fddb56ed78048fa8b90373c8a3cfc37b684831",
}


class STSBenchmarkSTS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STSBenchmark",
        dataset=_STS_BENCHMARK_DATASET,
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2021-01-01", "2021-12-31"),  # publication year
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
        superseded_by="STSBenchmark.v2",
    )

    min_score = 0
    max_score = 5


class STSBenchmarkSTSV2(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STSBenchmark.v2",
        dataset=_STS_BENCHMARK_DATASET,
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset. This version removes duplicate sentence pairs from the validation and test splits when the same order-insensitive pair appears more than once in an evaluation split or appears in another split.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2021-01-01", "2021-12-31"),  # publication year
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
        adapted_from=["STSBenchmark"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        def normalized_pair(row) -> tuple[str, str]:
            pair = (row["sentence1"], row["sentence2"])
            return pair if pair[0] <= pair[1] else (pair[1], pair[0])

        split_pair_counts: dict[tuple[str, str], dict[str, int]] = {}
        for split, dataset in self.dataset.items():
            for row in dataset:
                key = normalized_pair(row)
                split_pair_counts.setdefault(key, {})
                split_pair_counts[key][split] = split_pair_counts[key].get(split, 0) + 1

        eval_splits = ("validation", "test")
        for split in eval_splits:
            duplicate_eval_pairs = {
                key
                for key, counts in split_pair_counts.items()
                if split in counts and (counts[split] > 1 or len(counts) > 1)
            }
            self.dataset[split] = self.dataset[split].filter(
                lambda row: normalized_pair(row) not in duplicate_eval_pairs,
                num_proc=num_proc,
            )
