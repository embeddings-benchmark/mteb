from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RARbCode(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbCode",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b code-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "RAR-b/humanevalpack-mbpp-pooled",
            "revision": "25f7d11a7ac12dcbb8d3836eb2de682b98c825e4",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{muennighoff2023octopack,
  title={Octopack: Instruction tuning code large language models},
  author={Muennighoff, Niklas and Liu, Qian and Zebaze, Armel and Zheng, Qinkai and Hui, Binyuan and Zhuo, Terry Yue and Singh, Swayam and Tang, Xiangru and Von Werra, Leandro and Longpre, Shayne},
  journal={arXiv preprint arXiv:2308.07124},
  year={2023}
}
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
""",
        descriptive_stats={
            "n_samples": {"test": 1484},
            "avg_character_length": {
                "test": {
                    "average_document_length": 793.6813076734267,
                    "average_query_length": 375.7506738544474,
                    "num_documents": 301482,
                    "num_queries": 1484,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
