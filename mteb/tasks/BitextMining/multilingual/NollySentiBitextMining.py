from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = {
    "en-ha": ["eng-Latn", "hau-Latn"],
    "en-ig": ["eng-Latn", "ibo-Latn"],
    "en-pcm": ["eng-Latn", "pcm-Latn"],
    "en-yo": ["eng-Latn", "yor-Latn"],
}


class NollySentiBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="NollySentiBitextMining",
        dataset={
            "path": "gentaiscool/bitext_nollysenti_miners",
            "revision": "d48254fbdb51af1ae7f20831aab0bccf0b70a19c",
        },
        description="NollySenti is Nollywood movie reviews for five languages widely spoken in Nigeria (English, Hausa, Igbo, Nigerian-Pidgin, and Yoruba.",
        reference="https://huggingface.co/datasets/indonlp/nusatranslation_mt",
        type="BitextMining",
        category="s2s",
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2022-01-01", "2023-01-01"),
        form=["written"],
        domains=["Social", "Reviews"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{shode2023nollysenti,
            title={NollySenti: Leveraging Transfer Learning and Machine Translation for Nigerian Movie Sentiment Classification},
            author={Shode, Iyanuoluwa and Adelani, David Ifeoluwa and Peng, Jing and Feldman, Anna},
            booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
            pages={986--998},
            year={2023}
        }
        """,
        n_samples={"train": 1640},
        avg_character_length={"train": 4.46},
    )
