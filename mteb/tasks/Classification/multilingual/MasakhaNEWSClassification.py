from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = [
    "amh",
    "eng",
    "fra",
    "hau",
    "ibo",
    "lin",
    "lug",
    "orm",
    "pcm",
    "run",
    "sna",
    "som",
    "swa",
    "tir",
    "xho",
    "yor",
]


class MasakhaNEWSClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "MasakhaNEWSClassification",
            "hf_hub_name": "masakhane/masakhanews",
            "description": (
                "MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages."
            ),
            "reference": "https://arxiv.org/abs/2304.09972",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "8ccc72e69e65f40c70e117d8b3c08306bb788b60",
        }
