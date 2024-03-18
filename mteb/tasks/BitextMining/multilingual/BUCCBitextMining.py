from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = ["de-en", "fr-en", "ru-en", "zh-en"]


class BUCCBitextMining(AbsTaskBitextMining, CrosslingualTask):
    @property
    def description(self):
        return {
            "name": "BUCC",
            "hf_hub_name": "mteb/bucc-bitext-mining",
            "description": "BUCC bitext mining dataset",
            "reference": "https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": _LANGUAGES,
            "main_score": "f1",
            "revision": "d51519689f32196a32af33b075a01d0e7c51e252",
        }
