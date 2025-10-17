from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ChatDoctorRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChatDoctorRetrieval",
        description="A medical retrieval task based on ChatDoctor_HealthCareMagic dataset containing 112,000 real-world medical question-and-answer pairs. Each query is a medical question from patients (e.g., 'What are the symptoms of diabetes?'), and the corpus contains medical responses and healthcare information. The task is to retrieve the correct medical information that answers the patient's question. The dataset includes grammatical inconsistencies which help separate strong healthcare retrieval models from weak ones. Queries are patient medical questions while the corpus contains relevant medical responses, diagnoses, and treatment information from healthcare professionals.",
        reference="https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic",
        dataset={
            "path": "embedding-benchmark/ChatDoctor_HealthCareMagic",
            "revision": "50c2986fedffa33b38afd5c1752026f8e9e5ed1d",
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
        bibtex_citation=r"""
@misc{chatdoctor_healthcaremagic,
  title = {ChatDoctor HealthCareMagic: Medical Question-Answer Retrieval Dataset},
  url = {https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k},
  year = {2023},
}
""",
        prompt={
            "query": "Given a medical question from a patient, retrieve relevant healthcare information that best answers the question"
        },
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(
            self.metadata.dataset["path"],
            "corpus",
            revision=self.metadata.dataset["revision"],
        )["corpus"]
        queries_ds = load_dataset(
            self.metadata.dataset["path"],
            "queries",
            revision=self.metadata.dataset["revision"],
        )["queries"]
        qrels_ds = load_dataset(
            self.metadata.dataset["path"],
            "default",
            revision=self.metadata.dataset["revision"],
        )["test"]

        # Initialize data structures with 'test' split
        corpus = {}
        queries = {}
        relevant_docs = {}

        # Process corpus
        for item in corpus_ds:
            corpus[item["_id"]] = {"title": "", "text": item["text"]}

        # Process queries
        for item in queries_ds:
            queries[item["_id"]] = item["text"]

        # Process qrels (relevant documents)
        for item in qrels_ds:
            query_id = item["query-id"]
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][item["corpus-id"]] = int(item["score"])

        # Organize data by splits as expected by MTEB
        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": relevant_docs}

        self.data_loaded = True
