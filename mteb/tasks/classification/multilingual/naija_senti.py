from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NaijaSenti(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NaijaSenti",
        description="NaijaSenti is the first large-scale human-annotated Twitter sentiment dataset for the four most widely spoken languages in Nigeria — Hausa, Igbo, Nigerian-Pidgin, and Yorùbá — consisting of around 30,000 annotated tweets per language, including a significant fraction of code-mixed tweets.",
        reference="https://github.com/hausanlp/NaijaSenti",
        dataset={
            "path": "mteb/NaijaSenti",
            "revision": "c28be06e6ee80878a7398a0bb71b72c1969167ea",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "hau": ["hau-Latn"],
            "ibo": ["ibo-Latn"],
            "pcm": ["pcm-Latn"],
            "yor": ["yor-Latn"],
        },
        main_score="accuracy",
        date=("2022-05-01", "2023-05-08"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{muhammad-etal-2022-naijasenti,
  address = {Marseille, France},
  author = {Muhammad, Shamsuddeen Hassan  and
Adelani, David Ifeoluwa  and
Ruder, Sebastian  and
Ahmad, Ibrahim Sa{'}id  and
Abdulmumin, Idris  and
Bello, Bello Shehu  and
Choudhury, Monojit  and
Emezue, Chris Chinenye  and
Abdullahi, Saheed Salahudeen  and
Aremu, Anuoluwapo  and
Jorge, Al{\'\i}pio  and
Brazdil, Pavel},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  month = jun,
  pages = {590--602},
  publisher = {European Language Resources Association},
  title = {{N}aija{S}enti: A {N}igerian {T}witter Sentiment Corpus for Multilingual Sentiment Analysis},
  url = {https://aclanthology.org/2022.lrec-1.63},
  year = {2022},
}
""",
    )
