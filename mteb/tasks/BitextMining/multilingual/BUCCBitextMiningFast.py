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

class BUCCBitextMiningFast(AbsTaskBitextMining, CrosslingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="BUCC.v2",
        dataset={
            "path": "mteb/bucc-bitext-mining",
            "revision": "1739dc11ffe9b7bfccd7f3d585aeb4c544fc6677",
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