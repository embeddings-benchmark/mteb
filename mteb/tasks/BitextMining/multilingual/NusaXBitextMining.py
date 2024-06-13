from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = {
    "eng-ace": ["eng-Latn", "ace-Latn"],
    "eng-ban": ["eng-Latn", "ban-Latn"],
    "eng-bbc": ["eng-Latn", "bbc-Latn"],
    "eng-bjn": ["eng-Latn", "bjn-Latn"],
    "eng-bug": ["eng-Latn", "bug-Latn"],
    "eng-ind": ["eng-Latn", "ind-Latn"],
    "eng-jav": ["eng-Latn", "jav-Latn"],
    "eng-mad": ["eng-Latn", "mad-Latn"],
    "eng-min": ["eng-Latn", "min-Latn"],
    "eng-nij": ["eng-Latn", "nij-Latn"],
    "eng-sun": ["eng-Latn", "sun-Latn"],
}


class NusaXBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="NusaXBitextMining",
        dataset={
            "path": "gentaiscool/bitext_nusax",
            "revision": "fba4f2cfe2592641056f7a274c9aa8453b27a4a8",
        },
        description="NusaX is a parallel dataset for machine translation and sentiment analysis on 11 Indonesia languages and English.",
        reference="https://huggingface.co/datasets/indonlp/NusaX-senti/",
        type="BitextMining",
        category="s2s",
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
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
        @misc{winata2024miners,
            title={MINERS: Multilingual Language Models as Semantic Retrievers}, 
            author={Genta Indra Winata and Ruochen Zhang and David Ifeoluwa Adelani},
            year={2024},
            eprint={2406.07424},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        """,
        n_samples={"train": 5500},
        avg_character_length={"train": 5.00},
    )
