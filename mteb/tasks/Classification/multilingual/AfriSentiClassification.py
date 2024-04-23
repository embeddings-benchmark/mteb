from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class AfriSentiClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AfriSentiClassification",
        description="AfriSenti is the largest sentiment analysis dataset for under-represented African languages.",
        dataset={
            "path": "shmuhammad/AfriSenti-twitter-sentiment",
            "revision": "b52e930385cf5ed7f063072c3f7bd17b599a16cf",
        },
        reference="https://arxiv.org/abs/2302.08956",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "am": ["amh-Ethi"], # Amharic (Ethiopic script)
            "ar": ["ara-Arab"], # Moroccan Arabic, Standard Arabic (Arabic script)
            "ha": ["hau-Latn", "hau-Arab"], # Hausa (Latin script), additional script if written in Ajami (Arabic script)
            "ig": ["ibo-Latn"], # Igbo (Latin script)
            "rw": ["kin-Latn"], # Kinyarwanda (Latin script)
            "pt": ["por-Latn"], # Portuguese (Latin script)
            "pcm": ["pcm-Latn"], # Nigerian Pidgin (Latin script)
            "en": ["eng-Latn"], # English (Latin script)
            "ork": ["ork-Latn"], # Orokolo (Latin script)
            "sw": ["swa-Latn"], # Swahili (macrolanguage) (Latin script)
            "ti": ["tir-Ethi"], # Tigrinya (Ge'ez script)
            "tw": ["twi-Latn"], # Twi (Latin script)
            "ts": ["tso-Latn"], # Tsonga (Latin script)
            "yo": ["yor-Latn"] # Yoruba (Latin script)
            },
        main_score="accuracy",
        date=('2023-02-16', '2023-09-03'),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Topic classification"],
        license="Creative Commons Attribution 4.0 International License",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{Muhammad2023AfriSentiAT,
        title=AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages,
        author=Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino D'ario M'ario Ant'onio Ali and Davis Davis and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur,
        year=2023
        }""",
        n_samples={"test": 2048},
        avg_character_length=None,
    )
