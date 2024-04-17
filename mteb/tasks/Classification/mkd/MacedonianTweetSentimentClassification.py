from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MacedonianTweetSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MacedonianTweetSentimentClassification",
        description="An Macedonian dataset for tweet sentiment classification.",
        reference="https://aclanthology.org/R15-1034/",
        dataset={
            "path": "isaacchung/macedonian-tweet-sentiment-classification",
            "revision": "957e075ba35e4417ba7837987fd7053a6533a1a2",
        },
        type="Classification",
        category="s2s",
        date=["2014-11-01", "2015-04-01"],
        eval_splits=["test"],
        eval_langs=["mkd-Cyrl"],
        main_score="accuracy",
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC BY-NC-SA 3.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{jovanoski-etal-2015-sentiment,
    title = "Sentiment Analysis in {T}witter for {M}acedonian",
    author = "Jovanoski, Dame  and
      Pachovski, Veno  and
      Nakov, Preslav",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia  and
      Bontcheva, Kalina",
    booktitle = "Proceedings of the International Conference Recent Advances in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Hissar, Bulgaria",
    publisher = "INCOMA Ltd. Shoumen, BULGARIA",
    url = "https://aclanthology.org/R15-1034",
    pages = "249--257",
}""",
        n_samples={"test": 1139},
        avg_character_length={"test": 67.6},
    )
