from __future__ import annotations

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskRetrieval


class CQADupstackWordpressRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Wordpress-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-wordpress-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-wordpress-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackWebmastersRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Webmasters-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-webmasters-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-webmasters-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackUnixRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Unix-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-unix-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-unix-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackTexRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Tex-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-tex-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-tex-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackStatsRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Stats-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-stats-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-stats-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackProgrammersRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Programmers-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-programmers-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-programmers-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackPhysicsRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Physics-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-physics-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-physics-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackMathematicaRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Mathematica-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-mathematica-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-mathematica-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackGisRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Gis-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-gis-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-gis-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackGamingRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Gaming-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-gaming-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-gaming-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackEnglishRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-English-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-english-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-english-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )


class CQADupstackAndroidRetrievalPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstack-Android-PL",
        description="CQADupStack: A Stack Exchange Question Duplicate Pairs Dataset",
        reference="https://huggingface.co/datasets/clarin-knext/cqadupstack-android-pl",
        dataset={
            "path": "clarin-knext/cqadupstack-android-pl",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
    )
