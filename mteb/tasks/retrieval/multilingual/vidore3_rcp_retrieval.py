"""ViDoRe v3 with rubric-calibrated preference (RCP) gains.

Each task adapts an existing mteb task (see `adapted_from`): the same queries and corpus, plus a
candidate pool (`top_ranked`) and continuous relevance gains in the qrels' `gain` column.
Scored by `ndcg_float_at_10` (see `AbsTaskRetrievalFloatGains`). Metadata is copied from the
original task except for name, description, reference, dataset, eval_langs, main_score,
annotations_creators, citation and adapted_from.
"""

from __future__ import annotations

from mteb.abstasks.retrieval_float_gains import AbsTaskRetrievalFloatGains
from mteb.abstasks.task_metadata import TaskMetadata

_CITATION = r"""@misc{rcpndcg2026,
  title = {RCP-nDCG: Retrieval Evaluation with Rubric-Calibrated Preferences},
  year = {2026},
  note = {Preprint forthcoming},
}
"""


class Vidore3ComputerScienceRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "computer_science__english": ["eng-Latn"],
            "computer_science__french": ["fra-Latn"],
            "computer_science__german": ["deu-Latn"],
            "computer_science__italian": ["ita-Latn"],
            "computer_science__portuguese": ["por-Latn"],
            "computer_science__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Programming"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3ComputerScienceRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3EnergyRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3EnergyRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "energy__english": ["eng-Latn"],
            "energy__french": ["fra-Latn"],
            "energy__german": ["deu-Latn"],
            "energy__italian": ["ita-Latn"],
            "energy__portuguese": ["por-Latn"],
            "energy__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Chemistry", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3EnergyRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3FinanceEnRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnRCPRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "finance_en__english": ["eng-Latn"],
            "finance_en__french": ["fra-Latn"],
            "finance_en__german": ["deu-Latn"],
            "finance_en__italian": ["ita-Latn"],
            "finance_en__portuguese": ["por-Latn"],
            "finance_en__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3FinanceEnRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3FinanceFrRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrRCPRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "finance_fr__english": ["eng-Latn"],
            "finance_fr__french": ["fra-Latn"],
            "finance_fr__german": ["deu-Latn"],
            "finance_fr__italian": ["ita-Latn"],
            "finance_fr__portuguese": ["por-Latn"],
            "finance_fr__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3FinanceFrRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3HrRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3HrRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "hr__english": ["eng-Latn"],
            "hr__french": ["fra-Latn"],
            "hr__german": ["deu-Latn"],
            "hr__italian": ["ita-Latn"],
            "hr__portuguese": ["por-Latn"],
            "hr__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3HrRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3IndustrialRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3IndustrialRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "industrial__english": ["eng-Latn"],
            "industrial__french": ["fra-Latn"],
            "industrial__german": ["deu-Latn"],
            "industrial__italian": ["ita-Latn"],
            "industrial__portuguese": ["por-Latn"],
            "industrial__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3IndustrialRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3PharmaceuticalsRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "pharmaceuticals__english": ["eng-Latn"],
            "pharmaceuticals__french": ["fra-Latn"],
            "pharmaceuticals__german": ["deu-Latn"],
            "pharmaceuticals__italian": ["ita-Latn"],
            "pharmaceuticals__portuguese": ["por-Latn"],
            "pharmaceuticals__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Medical"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3PharmaceuticalsRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3PhysicsRCPRetrieval(AbsTaskRetrievalFloatGains):
    metadata = TaskMetadata(
        name="Vidore3PhysicsRCPRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models. Reranking over a 150-page candidate pool, scored with NDCG over continuous relevance gains (`ndcg_float_at_10`). Gains are rubric-calibrated preferences (RCP) from an LLM judge (Qwen3.5-397B-A17B), calibrated with a 2PL item-response model. The graded human qrels are unchanged. The main score averages the six language versions of each question; the paper reports the native-language questions. Gains come from judging the pages' OCR text.",
        reference="https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        dataset={
            "path": "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
            "revision": "b22323345cb5639944600d490969bcce9fba7943",
        },
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "physics__english": ["eng-Latn"],
            "physics__french": ["fra-Latn"],
            "physics__german": ["deu-Latn"],
            "physics__italian": ["ita-Latn"],
            "physics__portuguese": ["por-Latn"],
            "physics__spanish": ["spa-Latn"],
        },
        main_score="ndcg_float_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created and machine-translated",
        bibtex_citation=_CITATION
        + "\n@article{loison2026vidorev3comprehensiveevaluation,\n  archiveprefix = {arXiv},\n  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},\n  eprint = {2601.08620},\n  primaryclass = {cs.AI},\n  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},\n  url = {https://arxiv.org/abs/2601.08620},\n  year = {2026},\n}\n",
        adapted_from=["Vidore3PhysicsRetrieval.v2"],
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )
