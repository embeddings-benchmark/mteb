from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICK-E-PL",
        dataset={
            "path": "PL-MTEB/sicke-pl-pairclassification",
            "revision": "5c59e41555244b7e45c9a6be2d720ab4bafae558",
        },
        description="Polish version of SICK dataset for textual entailment.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
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
        n_samples=None,
        avg_character_length=None,
    )


class PpcPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PpcPC",
        dataset={
            "path": "PL-MTEB/ppc-pairclassification",
            "revision": "1.0",
        },
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
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
        n_samples=None,
        avg_character_length=None,
    )


class CdscePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CDSC-E",
        dataset={
            "path": "PL-MTEB/cdsce-pairclassification",
            "revision": "1.0",
        },
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
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
        n_samples=None,
        avg_character_length=None,
    )


class PscPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PSC",
        dataset={
            "path": "PL-MTEB/psc-pairclassification",
            "revision": "1.0",
        },
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
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
        n_samples=None,
        avg_character_length=None,
    )
