from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class AskUbuntuDupQuestions(AbsTaskReranking):
    metadata = TaskMetadata(
        name="AskUbuntuDupQuestions",
        description="AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "mteb/askubuntudupquestions-reranking",
            "revision": "2000358ca161889fa9c082cb41daa8dcfb161a54",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@article{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and  Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2104.06979",
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2104.06979",
}""",
        descriptive_stats={
            "n_samples": {"test": 2255},
            "test": {
                "average_document_length": 52.49722991689751,
                "average_query_length": 50.13019390581717,
                "num_documents": 7220,
                "num_queries": 361,
                "average_relevant_docs_per_query": 5.470914127423823,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 20.0,
            },
        },
    )
