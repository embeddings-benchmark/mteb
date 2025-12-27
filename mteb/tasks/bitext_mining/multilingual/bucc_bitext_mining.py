import logging

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "de-en": ["deu-Latn", "eng-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}


_SPLITS = ["test"]

logger = logging.getLogger(__name__)


class BUCCBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="BUCC",
        dataset={
            "path": "mteb/BUCC",
            "revision": "414572247440f0ccacf7eb0bb70a31533a0e5443",
        },
        description="BUCC bitext mining dataset train split.",
        reference="https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2017-01-01", "2018-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{zweigenbaum-etal-2017-overview,
  address = {Vancouver, Canada},
  author = {Zweigenbaum, Pierre  and
Sharoff, Serge  and
Rapp, Reinhard},
  booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
  doi = {10.18653/v1/W17-2512},
  editor = {Sharoff, Serge  and
Zweigenbaum, Pierre  and
Rapp, Reinhard},
  month = aug,
  pages = {60--67},
  publisher = {Association for Computational Linguistics},
  title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
  url = {https://aclanthology.org/W17-2512},
  year = {2017},
}
""",
        superseded_by="BUCC.v2",
    )

    def dataset_transform(self):
        dataset = {}
        for lang in self.dataset:
            dataset[lang] = {}
            for split in _SPLITS:
                data = self.dataset[lang][split]
                gold = data["gold"][0]
                gold = [(i - 1, j - 1) for (i, j) in gold]

                sentence1 = data["sentence1"][0]
                sentence2 = data["sentence2"][0]
                sentence1 = [
                    sentence1[i] for (i, j) in gold
                ]  # keep only sentences in gold. The 2nd value is meant for sentence2 but not used here. This is fixed in BUCC.v2.
                logger.info(f"Lang {lang} num gold {len(gold)}")
                logger.info(f"Lang {lang} num sentence1 {len(sentence1)}")
                logger.info(f"Lang {lang} num sentence2 {len(sentence2)}")
                dataset[lang][split] = {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "gold": gold,
                }
        self.dataset = dataset
