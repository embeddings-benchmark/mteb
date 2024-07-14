from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

_LANGUAGES = {
    "de-en": ["deu-Latn", "eng-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}


_SPLITS = ["test"]


class BUCCBitextMining(AbsTaskBitextMining, MultilingualTask):
    superseded_by = "BUCC.v2"
    metadata = TaskMetadata(
        name="BUCC",
        dataset={
            "path": "mteb/bucc-bitext-mining",
            "revision": "d51519689f32196a32af33b075a01d0e7c51e252",
        },
        description="BUCC bitext mining dataset",
        reference="https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2017-01-01", "2018-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="unknown",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation="""@inproceedings{zweigenbaum-etal-2017-overview,
    title = "Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora",
    author = "Zweigenbaum, Pierre  and
      Sharoff, Serge  and
      Rapp, Reinhard",
    editor = "Sharoff, Serge  and
      Zweigenbaum, Pierre  and
      Rapp, Reinhard",
    booktitle = "Proceedings of the 10th Workshop on Building and Using Comparable Corpora",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-2512",
    doi = "10.18653/v1/W17-2512",
    pages = "60--67",
    abstract = "This paper presents the BUCC 2017 shared task on parallel sentence extraction from comparable corpora. It recalls the design of the datasets, presents their final construction and statistics and the methods used to evaluate system results. 13 runs were submitted to the shared task by 4 teams, covering three of the four proposed language pairs: French-English (7 runs), German-English (3 runs), and Chinese-English (3 runs). The best F-scores as measured against the gold standard were 0.84 (German-English), 0.80 (French-English), and 0.43 (Chinese-English). Because of the design of the dataset, in which not all gold parallel sentence pairs are known, these are only minimum values. We examined manually a small sample of the false negative sentence pairs for the most precise French-English runs and estimated the number of parallel sentence pairs not yet in the provided gold standard. Adding them to the gold standard leads to revised estimates for the French-English F-scores of at most +1.5pt. This suggests that the BUCC 2017 datasets provide a reasonable approximate evaluation of the parallel sentence spotting task.",
}""",
        descriptive_stats={
            "n_samples": {"test": 641684},
            "avg_character_length": {"test": 101.3},
        },
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
                sentence1 = [sentence1[i] for (i, j) in gold]
                print(lang, len(gold))
                print(len(sentence1), len(sentence2))
                dataset[lang][split] = {
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "gold": gold,
                }
        self.dataset = dataset
