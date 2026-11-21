from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ChatDoctorRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChatDoctorRetrieval",
        description="A medical retrieval task based on ChatDoctor_HealthCareMagic dataset containing 112,000 real-world medical question-and-answer pairs. Each query is a medical question from patients (e.g., 'What are the symptoms of diabetes?'), and the corpus contains medical responses and healthcare information. The task is to retrieve the correct medical information that answers the patient's question. The dataset includes grammatical inconsistencies which help separate strong healthcare retrieval models from weak ones. Queries are patient medical questions while the corpus contains relevant medical responses, diagnoses, and treatment information from healthcare professionals.",
        reference="https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic",
        dataset={
            "path": "mteb/ChatDoctorRetrieval",
            "revision": "1774a6f1643a5d8b9781ff7a252651bbae2d93ba",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Medical"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Given a medical question from a patient, retrieve relevant healthcare information that best answers the question"
        },
    )
