from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RARbMath(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbMath",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "RAR-b/math-pooled",
            "revision": "2393603c0221ff52f448d12dd75f0856103c6cca",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2023-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval"],
        license="MIT",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{hendrycks2021measuring,
  title={Measuring mathematical problem solving with the math dataset},
  author={Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2103.03874},
  year={2021}
}
@article{cobbe2021training,
  title={Training verifiers to solve math word problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
@article{yu2023metamath,
  title={Metamath: Bootstrap your own mathematical questions for large language models},
  author={Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
  journal={arXiv preprint arXiv:2309.12284},
  year={2023}
}
""",
        n_samples={"test": 6319},
        avg_character_length={
            "test": {
                "average_document_length": 504.0197829347469,
                "average_query_length": 210.30732710871973,
                "num_documents": 389376,
                "num_queries": 6319,
                "average_relevant_docs_per_query": 1.0,
            }
        },
    )
