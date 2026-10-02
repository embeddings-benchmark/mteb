from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_TIERS = ["easy", "medium", "hard"]


class MathNetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MathNetRetrieval",
        description=(
            "The Math-Aware Retrieval benchmark from MathNet: given an Olympiad problem, retrieve a mathematically "
            "equivalent reformulation of it from a shared corpus of 117,088 documents containing equivalents at all "
            "difficulty tiers, LLM-generated near-miss hard negatives and distractors. Each of the 15,000 queries has "
            "exactly one gold equivalent per tier. The easy, medium and hard subsets increasingly disguise the surface "
            "form of the problem (from light paraphrase to heavy disguise with minimal lexical overlap) while "
            "preserving the underlying mathematics; the query's equivalents at the other tiers act as negatives. "
            "Equivalents and near-misses were generated with Gemini-3-Flash and kept only when both Gemini-3-Flash "
            "and GPT-5, used as LLM judges, agreed; a subset was additionally reviewed by humans. Problems are drawn "
            "from national and international olympiads; most text is English, with a small fraction also containing "
            "the original-language statement (e.g. Russian or Chinese)."
        ),
        reference="https://arxiv.org/abs/2604.18584",
        dataset={
            "path": "ShadenA/MathNet-Retrieve",
            "revision": "f3ecc756a7487c1a08d7a4bde1158d4091dbeabc",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={tier: ["eng-Latn"] for tier in _TIERS},
        main_score="ndcg_at_10",
        date=("1961-01-01", "2025-12-31"),
        domains=["Academic", "Written"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="multiple",  # queries are found olympiad problems, the corpus is LM-generated and LM-verified
        bibtex_citation=r"""
@inproceedings{alshammari2026mathnet,
  author = {Alshammari, Shaden and Wen, Kevin and Zainal, Abrar and Hamilton, Mark and Safaei, Navid and Albarakati, Sultan and Freeman, William T. and Torralba, Antonio},
  booktitle = {International Conference on Learning Representations},
  title = {MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval},
  url = {https://mathnet.mit.edu},
  year = {2026},
}
""",
        prompt={
            "query": "Given a math olympiad problem, retrieve a mathematically equivalent problem"
        },
    )
