from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = {
    "de-en": ["deu-Latn", "eng-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}


_SPLITS = ["test"]


class BUCCBitextMining(AbsTaskBitextMining, CrosslingualTask):
    superseeded_by = "BUCC.v2"
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
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 641684},
        avg_character_length={"test": 101.3},
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
