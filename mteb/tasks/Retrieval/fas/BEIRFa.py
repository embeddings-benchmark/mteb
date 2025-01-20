from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaFa(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="ArguAna-Fa",
        description="ArguAna-Fa",
        reference="https://huggingface.co/datasets/MCINext/arguana-fa",
        dataset={
            "path": "MCINext/arguana-fa",
            "revision": "bd449a14a4a7d5644ffd380a957c3c4bab83bf50",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )



class ClimateFEVERFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER-Fa",
        description="ClimateFEVER-Fa",
        reference="https://huggingface.co/datasets/MCINext/climate-fever-fa",
        dataset={
            "path": "MCINext/climate-fever-fa",
            "revision": "febfc57c1803a734cda52ebbc4f85a829d319b61",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class CQADupstackAndroidRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackAndroidRetrieval-Fa",
        description="CQADupstackAndroidRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-android-fa",
        dataset={
            "path": "MCINext/cqadupstack-android-fa",
            "revision": "1e828bfefc20aaf77b7c90d98f83cc3272f2ab9e",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackEnglishRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackEnglishRetrieval-Fa",
        description="CQADupstackEnglishRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-english-fa",
        dataset={
            "path": "MCINext/cqadupstack-english-fa",
            "revision": "d872fbb460dc3b6f927f7555ce2f7ee76a8b7073",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackGamingRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGamingRetrieval-Fa",
        description="CQADupstackGamingRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa",
        dataset={
            "path": "MCINext/cqadupstack-gaming-fa",
            "revision": "9cc3d417650243bf5f8e5424e33d0048e2cead81",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackGisRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGisRetrieval-Fa",
        description="CQADupstackGisRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa",
        dataset={
            "path": "MCINext/cqadupstack-gis-fa",
            "revision": "11a32a2de02cc377ceb65f5e9a2146017f783611",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackMathematicaRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackMathematicaRetrieval-Fa",
        description="CQADupstackMathematicaRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa",
        dataset={
            "path": "MCINext/cqadupstack-mathematica-fa",
            "revision": "2c4a0f77242290ff623edbced2848152d44f6b1a",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackPhysicsRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackPhysicsRetrieval-Fa",
        description="CQADupstackPhysicsRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa",
        dataset={
            "path": "MCINext/cqadupstack-physics-fa",
            "revision": "60b17339053fa23723841576a86e7f583e8b59c7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackProgrammersRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackProgrammersRetrieval-Fa",
        description="CQADupstackProgrammersRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa",
        dataset={
            "path": "MCINext/cqadupstack-programmers-fa",
            "revision": "0ff8ff35898efac6051db231434c298ec054f8b2",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackStatsRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackStatsRetrieval-Fa",
        description="CQADupstackStatsRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa",
        dataset={
            "path": "MCINext/cqadupstack-stats-fa",
            "revision": "c8a59accef7b5750ad03269f1017cf7a50ae449a",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackTexRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackTexRetrieval-Fa",
        description="CQADupstackTexRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa",
        dataset={
            "path": "MCINext/cqadupstack-tex-fa",
            "revision": "0d59dfaccac0764f53ccd8d63f8364dd27f5829a",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackUnixRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackUnixRetrieval-Fa",
        description="CQADupstackUnixRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa",
        dataset={
            "path": "MCINext/cqadupstack-unix-fa",
            "revision": "d136a58b44962cf25e8630b6887a7dee3916420e",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackWebmastersRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWebmastersRetrieval-Fa",
        description="CQADupstackWebmastersRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa",
        dataset={
            "path": "MCINext/cqadupstack-webmasters-fa",
            "revision": "8be7e5f57990c9fc7b2b7b859b9c0f8d704128e3",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class CQADupstackWordpressRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWordpressRetrieval-Fa",
        description="CQADupstackWordpressRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa",
        dataset={
            "path": "MCINext/cqadupstack-wordpress-fa",
            "revision": "c7e859b47542e8bad6846b93c71b9308f08c2b9d",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class DBPediaFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-Fa",
        description="DBPedia-Fa",
        reference="https://huggingface.co/datasets/MCINext/dbpedia-fa",
        dataset={
            "path": "MCINext/dbpedia-fa",
            "revision": "d3045f8bb952dab9173cc5067da595f2c28c2ca9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class FiQA2018Fa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018-Fa",
        description="FiQA2018-Fa",
        reference="https://huggingface.co/datasets/MCINext/fiqa-fa",
        dataset={
            "path": "MCINext/fiqa-fa",
            "revision": "a4f6b783977a81d44e53371be370c53c5b3bd9db",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class HotpotQAFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-Fa",
        description="HotpotQA-Fa",
        reference="https://huggingface.co/datasets/MCINext/hotpotqa-fa",
        dataset={
            "path": "MCINext/hotpotqa-fa",
            "revision": "d3369206326f66e102b13eecbe3bf1d670f99013",
        },

        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class MSMARCOFa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-Fa",
        description="MSMARCO-Fa",
        reference="https://huggingface.co/datasets/MCINext/msmarco-fa",
        dataset={
            "path": "MCINext/msmarco-fa",
            "revision": "9bd8f89f6f1f259734cddddb62dd468054fc77de",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class NFCorpusFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-Fa",
        description="NFCorpus-Fa",
        reference="https://huggingface.co/datasets/MCINext/nfcorpus-fa",
        dataset={
            "path": "MCINext/nfcorpus-fa",
            "revision": "a7873741b8f5f88f84114c91861e3644baaa4b1c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class NQFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-Fa",
        description="NQ-Fa",
        reference="https://huggingface.co/datasets/MCINext/nq-fa",
        dataset={
            "path": "MCINext/nq-fa",
            "revision": "69dcd4788696ecae180f8b835026498250853017",
        },

        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class QuoraRetrievalFa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrieval-Fa",
        description="QuoraRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/quora-fa",
        dataset={
            "path": "MCINext/quora-fa",
            "revision": "bdf1618da81ab481884ab0358ec502ebc093e6c9",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class SCIDOCSFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-Fa",
        description="SCIDOCS-Fa",
        reference="https://huggingface.co/datasets/MCINext/scidocs-fa",
        dataset={
            "path": "MCINext/scidocs-fa",
            "revision": "7d9d5a782304f7ca396e76ab738eac762812b9ab",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class SciFactFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-Fa",
        description="SciFact-Fa",
        reference="https://huggingface.co/datasets/MCINext/scifact-fa",
        dataset={
            "path": "MCINext/scifact-fa",
            "revision": "939778ed248ba84d25bf05f8b3d38ed3e371ae38",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )

class TRECCOVIDFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-Fa",
        description="TRECCOVID-Fa",
        reference="https://huggingface.co/datasets/MCINext/trec-covid-fa",
        dataset={
            "path": "MCINext/trec-covid-fa",
            "revision": "e52ad327135547992463c230d9eb97a13439ab9d",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )


class Touche2020Fa(AbsTaskRetrieval):

    metadata = TaskMetadata(
        name="Touche2020-Fa",
        description="Touche2020-Fa",
        reference="https://huggingface.co/datasets/MCINext/touche2020-fa",
        dataset={
            "path": "MCINext/touche2020-fa",
            "revision": "b7b9a535a1a62791aeaef666b0ee69d34c891335",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )