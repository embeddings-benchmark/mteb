from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = [
    "eng-ace": ["eng","ace"],
    "eng-ban": ["eng","ban"],
    "eng-bbc": ["eng","bbc"],
    "eng-bjn": ["eng","bjn"],
    "eng-bug": ["eng","bug"],
    "eng-ind": ["eng","ind"],
    "eng-jav": ["eng","jav"],
    "eng-mad": ["eng","mad"],
    "eng-min": ["eng","min"],
    "eng-nij": ["eng","nij"],
    "eng-sun": ["eng","sun"]
]

class NusaXBitextMining(AbsTaskBitextMining, CrosslingualTask):
    parallel_subsets = True
    metadata = TaskMetadata(
        name="NusaXBitextMining",
        dataset={
            "path": "gentaiscool/bitext_nusax",
            "revision": "01feaeded353a31a27e010b7818bc9a3db82af40",
            "trust_remote_code": True,
        },
        description="NusaX is a parallel dataset for machine translation and sentiment analysis on 11 Indonesia languages and English.",
        reference="https://huggingface.co/datasets/facebook/flores",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2023-01-01"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
        @inproceedings{winata2023nusax,
        title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
        author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and others},
        booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
        pages={815--834},
        year={2023}
        }
        """,
        n_samples={"test":5500},
        avg_character_length={},
    )
