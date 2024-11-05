from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "McGill-NLP/WebLINX",
            "name": "reranking",
            "revision": "ed1c933c2b3617e5700d8a7ebe07f5975969a453",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["eng-Latn"],
        main_score="mrr_at_10",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""
@misc{lù2024weblinx,
      title={WebLINX: Real-World Website Navigation with Multi-Turn Dialogue}, 
      author={Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      year={2024},
      eprint={2402.05930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
        """,
        descriptive_stats={
            "n_samples": {
                "validation": 1301,
                "test_iid": 1438,
                "test_cat": 3560,
                "test_web": 3144,
                "test_vis": 5298,
                "test_geo": 4916,
            },
            "validation": {
                "average_document_length": 318.17634941296905,
                "average_query_length": 1647.5180630284397,
                "num_documents": 316508,
                "num_queries": 1301,
                "average_relevant_docs_per_query": 1.01076095311299,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 243.2805534204458,
            },
            "test_iid": {
                "average_document_length": 318.135696550501,
                "average_query_length": 1722.6321279554938,
                "num_documents": 405972,
                "num_queries": 1438,
                "average_relevant_docs_per_query": 1.0528511821974966,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 282.317107093185,
            },
            "test_cat": {
                "average_document_length": 313.91351392594606,
                "average_query_length": 2149.6587078651687,
                "num_documents": 1258191,
                "num_queries": 3560,
                "average_relevant_docs_per_query": 1.0016853932584269,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 353.4244382022472,
            },
            "test_geo": {
                "average_document_length": 315.00053963351843,
                "average_query_length": 1742.6588689991863,
                "num_documents": 1150781,
                "num_queries": 4916,
                "average_relevant_docs_per_query": 1.0024410089503661,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 234.08889340927584,
            },
            "test_vis": {
                "average_document_length": 327.165126601106,
                "average_query_length": 1737.2595318988297,
                "num_documents": 1606858,
                "num_queries": 5298,
                "average_relevant_docs_per_query": 1.0152887882219706,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 303.2952057380143,
            },
            "test_web": {
                "average_document_length": 326.280188209908,
                "average_query_length": 1831.4624681933842,
                "num_documents": 834175,
                "num_queries": 3144,
                "average_relevant_docs_per_query": 1.0588422391857506,
                "average_instruction_length": 0,
                "num_instructions": 0,
                "average_top_ranked_per_query": 265.3228371501272,
            },
        },
    )
