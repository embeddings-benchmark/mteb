from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "nld_Latn": "nl",
    "srn_Latn": "srn",
}
_SPLIT = ["test"]

# number of sentences to use for evaluation (256 is full test set)
_N = 256

_EVAL_LANGS = {
    "srn_Latn-nld_Latn": ["srn-Latn", "nld-Latn"],
    "nld_Latn-srn_Latn": ["nld-Latn", "srn-Latn"],
}


class SRNCorpusBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="SRNCorpusBitextMining",
        dataset={
            "path": "davidstap/sranantongo",
            "revision": "2903226ff89ca0b15221a75d32b6355248295119",
            "trust_remote_code": True,
        },
        description="SRNCorpus is a machine translation corpus for creole language Sranantongo and Dutch.",
        reference="https://arxiv.org/abs/2212.06383",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=("2022-04-01", "2022-07-31"),
        form=["written"],
        domains=["Social", "Web"],
        task_subtypes=[],
        license="CC-BY-SA-4.0",
        socioeconomic_status="medium",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        n_samples={"test": _N},
        avg_character_length={"test": 55},
        bibtex_citation="""
@article{zwennicker2022towards,
  title={Towards a general purpose machine translation system for Sranantongo},
  author={Zwennicker, Just and Stap, David},
  journal={arXiv preprint arXiv:2212.06383},
  year={2022}
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}

        def _clean_columns(batch, keys):
            """Clean dataset features"""
            return {key: [s.strip("\r") for s in batch[key]] for key in keys}

        for lang in self.langs:
            l1, l2 = lang.split("-")
            dataset = datasets.load_dataset(
                name="srn-nl_other",
                split="test",
                **self.metadata_dict["dataset"],
            ).map(lambda batch: _clean_columns(batch, ["nl", "srn"]), batched=True)
            dataset = dataset.rename_columns(
                {_LANGUAGES[l1]: "sentence1", _LANGUAGES[l2]: "sentence2"}
            )
            self.dataset[lang] = datasets.DatasetDict({"test": dataset})

        self.data_loaded = True
