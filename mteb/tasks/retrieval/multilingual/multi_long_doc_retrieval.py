from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Hang"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "zh": ["cmn-Hans"],
}


class MultiLongDocRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiLongDocRetrieval",
        description="Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024). It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.",
        reference="https://arxiv.org/abs/2402.03216",  # also: https://huggingface.co/datasets/Shitao/MLDR
        dataset={
            "path": "mteb/MultiLongDocRetrieval",
            "revision": "837028901907a7d419b4ab906f28e011ce1cc824",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=(
            "2000-01-01",
            "2024-12-31",
        ),  # Not found in the paper, guessed using the paper's publication date and constituent datasets
        domains=[
            "Encyclopaedic",
            "Written",
            "Web",
            "Non-fiction",
            "Fiction",
        ],  # narrativeqa, wikipedia, wudao, mC4
        task_subtypes=[],
        license="mit",
        annotations_creators="LM-generated",  # gpt-3.5
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{bge-m3,
  archiveprefix = {arXiv},
  author = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
  eprint = {2402.03216},
  primaryclass = {cs.CL},
  title = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  year = {2024},
}
""",
    )
