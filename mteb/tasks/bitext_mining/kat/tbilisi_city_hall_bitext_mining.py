from datasets import DatasetDict, load_dataset

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "kat_Geor": "ka",
    "eng_Latn": "en",
}

_EVAL_LANGS = {
    "kat_Geor-eng_Latn": ["kat-Geor", "eng-Latn"],
    "eng_Latn-kat_Geor": ["eng-Latn", "kat-Geor"],
}
_EVAL_SPLIT = "test"


class TbilisiCityHallBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="TbilisiCityHallBitextMining",
        dataset={
            "path": "jupyterjazz/tbilisi-city-hall-titles",
            "revision": "798bb599140565cca2dab8473035fa167e5ee602",
        },
        description="Parallel news titles from the Tbilisi City Hall website (https://tbilisi.gov.ge/).",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        domains=["News", "Written"],
        sample_creation="created",
        reference="https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles",
        date=("2024-05-02", "2024-05-03"),
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {}

        for lang in self.hf_subsets:
            l1, l2 = lang.split("-")
            dataset = load_dataset(
                self.metadata.dataset["path"],
                split=_EVAL_SPLIT,
                revision=self.metadata.dataset["revision"],
            )
            dataset = dataset.rename_columns(
                {_LANGUAGES[l1]: "sentence1", _LANGUAGES[l2]: "sentence2"}
            )
            self.dataset[lang] = DatasetDict({_EVAL_SPLIT: dataset})

        self.data_loaded = True
