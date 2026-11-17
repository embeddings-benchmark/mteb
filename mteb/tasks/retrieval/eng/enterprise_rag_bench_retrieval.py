from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EnterpriseRAGBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EnterpriseRAGBenchRetrieval",
        description="Retrieval over a synthetic company's internal knowledge from EnterpriseRAG-Bench: Slack messages, emails, project and support tickets, shared files, CRM records, meeting transcripts, pull requests and wiki pages. Queries are questions an employee would ask, from single-document lookups to questions that require several related or conflicting documents, and the relevant documents are the gold documents that answer them. The 30 questions without gold documents (high-level and info-not-found) are left out.",
        reference="https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench",
        dataset={
            "path": "sinhal/EnterpriseRAGBenchRetrieval",
            "revision": "631516a0b65afb9091fa1e4b1c60bd16e50707d4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-01-01", "2026-05-05"),
        domains=["Written", "Engineering"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        prompt={
            "query": "Given a question about a company's internal work, retrieve the internal documents that answer it"
        },
        bibtex_citation=r"""
@misc{sun2026enterpriseragbench,
  archiveprefix = {arXiv},
  author = {Sun, Yuhong and Rahmfeld, Joachim and Weaver, Chris and Chen, Weijia and Desai, Roshan and Huang, Wenxi and Butler, Mark H.},
  eprint = {2605.05253},
  primaryclass = {cs.IR},
  title = {{EnterpriseRAG-Bench}: A {RAG} Benchmark for Company Internal Knowledge},
  url = {https://arxiv.org/abs/2605.05253},
  year = {2026},
}
""",
    )
