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
        revision="71bba34b0ece6c56dfcf46d9758a27f7a90f17e9",
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
        hf_hub_name="PL-MTEB/ppc-pairclassification",
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="2c7d2df57801a591f6b1e3aaf042e7a04ec7d9f2",
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
        hf_hub_name="PL-MTEB/cdsce-pairclassification",
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="0a3d4aa409b22f80eb22cbf59b492637637b536d",
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
        hf_hub_name="PL-MTEB/psc-pairclassification",
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="ap",
        revision="d05a294af9e1d3ff2bfb6b714e08a24a6cabc669",
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
