from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NoRecSentenceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NoRecSentenceClassification",
        description="Sentence-level polarity classification of Norwegian sentences from professional reviews across mixed domains (literature, movies, games, restaurants, music, theater and products). The sentence labels are aggregated from the fine-grained annotations of NoReC_fine. This task uses the binary version, where each sentence is positive or negative.",
        reference="https://huggingface.co/datasets/ltg/norec_sentence",
        dataset={
            "path": "Avalyset/NoRecSentenceClassification",
            "revision": "2407bfe4181d05f96e3ef6a5fabfc78a68a67dd3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="accuracy",
        date=("1998-01-01", "2019-12-31"),
        domains=["Written", "Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kutuzov-etal-2021-large,
  address = {Reykjavik, Iceland (Online)},
  archiveprefix = {arXiv},
  author = {Kutuzov, Andrey  and
Barnes, Jeremy  and
Velldal, Erik  and
{\O}vrelid, Lilja  and
Oepen, Stephan},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  eprint = {2104.06546},
  title = {Large-Scale Contextualised Language Modelling for {N}orwegian},
  url = {https://aclanthology.org/2021.nodalida-main.4},
  year = {2021},
}

@inproceedings{ovrelid-etal-2020-fine,
  address = {Marseille, France},
  archiveprefix = {arXiv},
  author = {{\O}vrelid, Lilja  and
M{\ae}hlum, Petter  and
Barnes, Jeremy  and
Velldal, Erik},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  eprint = {1911.12722},
  title = {A Fine-grained Sentiment Dataset for {N}orwegian},
  url = {https://aclanthology.org/2020.lrec-1.618},
  year = {2020},
}
""",
        prompt="Classify Norwegian sentences from reviews by sentiment",
    )

    label_column_name = "sentiment"
    input_column_name = "review"
    samples_per_label = 32
