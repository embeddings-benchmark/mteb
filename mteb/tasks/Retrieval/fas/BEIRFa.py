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
            "revision": "fa97814be356fe4d18caadb457b4469bd34019ca",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Blog"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["ArguAna"],
    )


class ClimateFEVERFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER-Fa",
        description="ClimateFEVER-Fa",
        reference="https://huggingface.co/datasets/MCINext/climate-fever-fa",
        dataset={
            "path": "MCINext/climate-fever-fa",
            "revision": "45d9176b6fcba33abc58494ee82f18ee7e8ddbae",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["ClimateFEVER"],
    )


class CQADupstackAndroidRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackAndroidRetrieval-Fa",
        description="CQADupstackAndroidRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-android-fa",
        dataset={
            "path": "MCINext/cqadupstack-android-fa",
            "revision": "bcdaf4e30477eea9b9b17ecbb153ca403e5c3ebd",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackAndroid"],
    )


class CQADupstackEnglishRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackEnglishRetrieval-Fa",
        description="CQADupstackEnglishRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-english-fa",
        dataset={
            "path": "MCINext/cqadupstack-english-fa",
            "revision": "029a2e69e7d9e68b6bdc471073606104417a5be7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackEnglish"],
    )


class CQADupstackGamingRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGamingRetrieval-Fa",
        description="CQADupstackGamingRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-gaming-fa",
        dataset={
            "path": "MCINext/cqadupstack-gaming-fa",
            "revision": "e9c7ad03f29a55ab14eae730146961b8cdc14227",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackGamingRetrieval"],
    )


class CQADupstackGisRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackGisRetrieval-Fa",
        description="CQADupstackGisRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-gis-fa",
        dataset={
            "path": "MCINext/cqadupstack-gis-fa",
            "revision": "e907c4144dc27bc8a035d78d69e15f39c56623a9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackGisRetrieval"],
    )


class CQADupstackMathematicaRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackMathematicaRetrieval-Fa",
        description="CQADupstackMathematicaRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-mathematica-fa",
        dataset={
            "path": "MCINext/cqadupstack-mathematica-fa",
            "revision": "b92e24fab42ab599535a19ee744de5485ec92f64",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackMathematicaRetrieval"],
    )


class CQADupstackPhysicsRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackPhysicsRetrieval-Fa",
        description="CQADupstackPhysicsRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-physics-fa",
        dataset={
            "path": "MCINext/cqadupstack-physics-fa",
            "revision": "816ad7473d6813f77a0ca5e72b1ff7a52752d370",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackPhysicsRetrieval"],
    )


class CQADupstackProgrammersRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackProgrammersRetrieval-Fa",
        description="CQADupstackProgrammersRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-programmers-fa",
        dataset={
            "path": "MCINext/cqadupstack-programmers-fa",
            "revision": "be6460df57ab7c1b2c9fe295a31660dbd077ecf0",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackProgrammersRetrieval"],
    )


class CQADupstackStatsRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackStatsRetrieval-Fa",
        description="CQADupstackStatsRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-stats-fa",
        dataset={
            "path": "MCINext/cqadupstack-stats-fa",
            "revision": "c6e2c8b6153958118ec04352dd82a30ea2e2cad5",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackStatsRetrieval"],
    )


class CQADupstackTexRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackTexRetrieval-Fa",
        description="CQADupstackTexRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-tex-fa",
        dataset={
            "path": "MCINext/cqadupstack-tex-fa",
            "revision": "860d152c86fda27229270b6bf4e832ff374ac01b",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackTexRetrieval"],
    )


class CQADupstackUnixRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackUnixRetrieval-Fa",
        description="CQADupstackUnixRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-unix-fa",
        dataset={
            "path": "MCINext/cqadupstack-unix-fa",
            "revision": "c2a326387954aad66ff00d324a9278067b1e3bb6",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackUnixRetrieval"],
    )


class CQADupstackWebmastersRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWebmastersRetrieval-Fa",
        description="CQADupstackWebmastersRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-webmasters-fa",
        dataset={
            "path": "MCINext/cqadupstack-webmasters-fa",
            "revision": "506f29f8ce59648efe99afee736b0b158eced516",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackWebmasters"],
    )


class CQADupstackWordpressRetrievalFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackWordpressRetrieval-Fa",
        description="CQADupstackWordpressRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/cqadupstack-wordpress-fa",
        dataset={
            "path": "MCINext/cqadupstack-wordpress-fa",
            "revision": "7f755e88647b4023df52da04d4e3d31a7de5fcb0",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["CQADupstackWordpressRetrieval"],
    )


class DBPediaFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-Fa",
        description="DBPedia-Fa",
        reference="https://huggingface.co/datasets/MCINext/dbpedia-fa",
        dataset={
            "path": "MCINext/dbpedia-fa",
            "revision": "13529e6e301e9d72f86def882cfbca04791d83f9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["DBPedia"],
    )


class FiQA2018Fa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018-Fa",
        description="FiQA2018-Fa",
        reference="https://huggingface.co/datasets/MCINext/fiqa-fa",
        dataset={
            "path": "MCINext/fiqa-fa",
            "revision": "e683ce7ecd0b47edc3e29fda7cfd75335be4a997",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["FiQA2018"],
    )


class HotpotQAFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-Fa",
        description="HotpotQA-Fa",
        reference="https://huggingface.co/datasets/MCINext/hotpotqa-fa",
        dataset={
            "path": "MCINext/hotpotqa-fa",
            "revision": "1cafde1306aa56b5dfdce0b14633ae9ee1a63ddb",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["HotpotQA"],
    )


class MSMARCOFa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-Fa",
        description="MSMARCO-Fa",
        reference="https://huggingface.co/datasets/MCINext/msmarco-fa",
        dataset={
            "path": "MCINext/msmarco-fa",
            "revision": "88f90b0b04f91778ba5341095b0a9f1d7799ce10",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["MSMARCO"],
    )


class NFCorpusFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus-Fa",
        description="NFCorpus-Fa",
        reference="https://huggingface.co/datasets/MCINext/nfcorpus-fa",
        dataset={
            "path": "MCINext/nfcorpus-fa",
            "revision": "70aa71825a791e87210c0355a01f538aa611feae",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["NFCorpus"],
    )


class NQFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-Fa",
        description="NQ-Fa",
        reference="https://huggingface.co/datasets/MCINext/nq-fa",
        dataset={
            "path": "MCINext/nq-fa",
            "revision": "d4ea898b644c8d5f608b60947cb637bebbf1ac66",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["NQ"],
    )


class QuoraRetrievalFa(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrieval-Fa",
        description="QuoraRetrieval-Fa",
        reference="https://huggingface.co/datasets/MCINext/quora-fa",
        dataset={
            "path": "MCINext/quora-fa",
            "revision": "1a43f4f5dcd71e6b14b275ae82c3237cdd4fd5fd",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["QuoraRetrieval"],
    )


class SCIDOCSFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-Fa",
        description="SCIDOCS-Fa",
        reference="https://huggingface.co/datasets/MCINext/scidocs-fa",
        dataset={
            "path": "MCINext/scidocs-fa",
            "revision": "6611ebf4b4c1aaf8b021e4728440db2188291b8b",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["SCIDOCS"],
    )


class SciFactFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-Fa",
        description="SciFact-Fa",
        reference="https://huggingface.co/datasets/MCINext/scifact-fa",
        dataset={
            "path": "MCINext/scifact-fa",
            "revision": "7723397096199c4d6f367b445fccaf282c518abe",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["SciFact"],
    )


class TRECCOVIDFa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-Fa",
        description="TRECCOVID-Fa",
        reference="https://huggingface.co/datasets/MCINext/trec-covid-fa",
        dataset={
            "path": "MCINext/trec-covid-fa",
            "revision": "98e6c2d33dfa166ee326e8b1bc7ea82c7e6898dd",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Medical"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["TRECCOVID"],
    )


class Touche2020Fa(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020-Fa",
        description="Touche2020-Fa",
        reference="https://huggingface.co/datasets/MCINext/touche2020-fa",
        dataset={
            "path": "MCINext/touche2020-fa",
            "revision": "0f464636f91641cc6ef6f6f8f249c73f4a609982",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["Touche2020"],
    )
