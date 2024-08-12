from __future__ import annotations
import os
import logging

from mteb.abstasks.TaskMetadata import TaskMetadata


from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)


class ChemNQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemNQRetrieval",
        dataset={
            "path": "BASF-We-Create-Chemistry/ChemNQRetrieval",
            "revision": "023e7a813e3b73d8d33551ed2aea511314d612e2",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@article{47761,title	= {Natural Questions: a Benchmark for Question Answering Research},
        author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh 
        and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee 
        and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le 
        and Slav Petrov},year	= {2019},journal	= {Transactions of the Association of Computational 
        Linguistics}}""",
        descriptive_stats={}
    )
