from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class TwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterURLCorpus",
        dataset={
            "path": "mteb/twitterurlcorpus-pairclassification",
            "revision": "8b6510b0b1fa4e4c4f879467980e9be563ec1cdf",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 51534},
        avg_character_length={"test": 79.5},
    )
