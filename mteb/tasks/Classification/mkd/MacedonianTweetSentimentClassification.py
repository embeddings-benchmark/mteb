from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
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
        modalities=["text"],
        date=["2014-11-01", "2015-04-01"],
        eval_splits=["test"],
        eval_langs=["mkd-Cyrl"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{jovanoski-etal-2015-sentiment,
  address = {Hissar, Bulgaria},
  author = {Jovanoski, Dame  and
Pachovski, Veno  and
Nakov, Preslav},
  booktitle = {Proceedings of the International Conference Recent Advances in Natural Language Processing},
  editor = {Mitkov, Ruslan  and
Angelova, Galia  and
Bontcheva, Kalina},
  month = sep,
  pages = {249--257},
  publisher = {INCOMA Ltd. Shoumen, BULGARIA},
  title = {Sentiment Analysis in {T}witter for {M}acedonian},
  url = {https://aclanthology.org/R15-1034},
  year = {2015},
}
""",
    )
