from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SWEPolyBenchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SWEPolyBenchRetrieval",
        description="A repository-level code retrieval task based on SWE-PolyBench, a multi-language benchmark of real GitHub issues from JavaScript, TypeScript, Python and Java repositories. Each query is a GitHub issue description and is evaluated only against the files of its own repository snapshot (provided through the top_ranked subset), which together form the corpus. The relevant documents are the files modified by the reference solution.",
        reference="https://huggingface.co/datasets/AmazonScience/SWE-PolyBench",
        dataset={
            "path": "mteb/SWEPolyBenchRetrieval",
            "revision": "8df29658f9486935e26c7672ed880762b533b66a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[
            "eng-Latn",
            "javascript-Code",
            "typescript-Code",
            "python-Code",
            "java-Code",
        ],
        main_score="ndcg_at_10",
        date=("2016-01-11", "2024-10-18"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a GitHub issue, retrieve the repository files that need to be modified to resolve the issue."
        },
        bibtex_citation=r"""
@misc{rashid2025swepolybenchmultilanguagebenchmarkrepository,
  archiveprefix = {arXiv},
  author = {Muhammad Shihab Rashid and Christian Bock and Yuan Zhuang and Alexander Buchholz and Tim Esler and Simon Valentin and Luca Franceschi and Martin Wistuba and Prabhu Teja Sivaprasad and Woo Jung Kim and Anoop Deoras and Giovanni Zappella and Laurent Callot},
  eprint = {2504.08703},
  primaryclass = {cs.SE},
  title = {SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents},
  url = {https://arxiv.org/abs/2504.08703},
  year = {2025},
}
""",
    )
