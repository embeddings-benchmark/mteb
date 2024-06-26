from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQ(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ",
        dataset={
            "path": "mteb/nq",
            "revision": "b774495ed302d8c44a3a7ea25c90dbce03968f31",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@article{47761,title	= {Natural Questions: a Benchmark for Question Answering Research},
        author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh 
        and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee 
        and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le 
        and Slav Petrov},year	= {2019},journal	= {Transactions of the Association of Computational 
        Linguistics}}""",
        n_samples=None,
        avg_character_length={
            "test": {
                "average_document_length": 492.2287851281462,
                "average_query_length": 48.17902665121669,
                "num_documents": 2681468,
                "num_queries": 3452,
                "average_relevant_docs_per_query": 1.2169756662804172,
            }
        },
    )
