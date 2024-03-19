import datasets

from ....abstasks import AbsTaskSTS, MultilingualTask

_LANGUAGES = ["en", "de", "es", "fr", "it", "nl", "pl", "pt", "ru", "zh"]
_SPLITS = ["dev", "test"]


class STSBenchmarkMultilingualSTS(AbsTaskSTS, MultilingualTask):
    @property
    def metadata_dict(self):
        return {
            "name": "STSBenchmarkMultilingualSTS",
            "hf_hub_name": "stsb_multi_mt",
            "description": (
                "Semantic Textual Similarity Benchmark (STSbenchmark) dataset,"
                "but translated using DeepL API."
            ),
            "reference": "https://github.com/PhilipMay/stsb-multi-mt/",
            "type": "STS",
            "category": "s2s",
            "eval_splits": _SPLITS,
            "eval_langs": _LANGUAGES,
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "93d57ef91790589e3ce9c365164337a8a78b7632",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        def get_dataset_subset(lang: str):
            """For a specified subset (=language)
            only get the splits listed in _SPLIT
            and rename column "score"

            Args:
                lang (str): _description_

            Returns:
                datasets.DatasetDict: the dataset of the specified language
            """
            subset = datasets.DatasetDict(
                **dict(
                    zip(
                        _SPLITS,
                        datasets.load_dataset(
                            self.metadata_dict["hf_hub_name"],
                            lang,
                            split=_SPLITS,
                            revision=self.metadata_dict.get("revision", None),
                        ),
                    )
                )
            )
            return subset.rename_column("similarity_score", "score")

        self.dataset = datasets.DatasetDict(
            **dict(zip(self.langs, [get_dataset_subset(lang) for lang in self.langs]))
        )

        self.data_loaded = True
