from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICK-E-PL",
        hf_hub_name="PL-MTEB/sicke-pl-pairclassification",
        description="Polish version of SICK dataset for textual entailment.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="5c59e41555244b7e45c9a6be2d720ab4bafae558",
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
    )



class PpcPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PpcPC",
        hf_hub_name="PL-MTEB/ppc-pairclassification",
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="1.0",
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
    )


class CdscePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CDSC-E",
        hf_hub_name="PL-MTEB/cdsce-pairclassification",
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="1.0",
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
    )



class PscPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PSC",
        hf_hub_name="PL-MTEB/psc-pairclassification",
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="1.0",
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
    )

