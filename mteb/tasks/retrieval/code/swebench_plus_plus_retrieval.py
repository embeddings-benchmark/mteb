from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEBenchPlusPlusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEBenchPlusPlusRetrieval",
        description="A repository-level code retrieval task based on SWE-bench++, a set of real GitHub issues and pull requests from diverse open-source repositories in Python, Go, Java, TypeScript, JavaScript, Rust and C++. Each query is a GitHub issue description and is evaluated only against the files of its own repository snapshot (provided through the top_ranked subset), which together form the corpus. The relevant documents are the files modified by the reference solution.",
        reference="https://huggingface.co/datasets/TuringEnterprises/SWE-Bench-plus-plus",
        dataset={
            "path": "mteb/SWEBenchPlusPlusRetrieval",
            "revision": "19a7fc0d722dad5db0c704c0a060e8e345e6c093",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[
            "eng-Latn",
            "python-Code",
            "go-Code",
            "java-Code",
            "typescript-Code",
            "javascript-Code",
            "rust-Code",
            "c++-Code",
        ],
        main_score="ndcg_at_10",
        date=("2020-05-24", "2025-09-18"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="https://huggingface.co/datasets/TuringEnterprises/SWE-Bench-plus-plus#7-licensing-and-permissions",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a GitHub issue, retrieve the repository files that need to be modified to resolve the issue."
        },
        bibtex_citation=r"""
@misc{wang2025swebenchplusplus,
  archiveprefix = {arXiv},
  author = {Lilin Wang and Lucas Ramalho and Alan Celestino and Phuc Anthony Pham and Yu Liu and Umang Kumar Sinha and Andres Portillo and Onassis Osunwa and Gabriel Maduekwe},
  eprint = {2512.17419},
  primaryclass = {cs.SE},
  title = {{SWE-Bench++}: A Framework for the Scalable Generation of Software Engineering Benchmarks from Open-Source Repositories},
  url = {https://arxiv.org/abs/2512.17419},
  year = {2025},
}
""",
    )
