from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
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
        superseded_by="ArguAna-Fa.v2",
    )


class ArguAnaFaV2(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="ArguAna-Fa.v2",
        description="ArguAna-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/arguana-fa-v2",
        dataset={
            "path": "MCINext/arguana-fa-v2",
            "revision": "3c742a98c0c17f7041acf0895680e0ec0000786f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        category="t2t",
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
        adapted_from=["CQADupstackAndroidRetrieval"],
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
        category="t2t",
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
        adapted_from=["CQADupstackEnglishRetrieval"],
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        adapted_from=["CQADupstackWebmastersRetrieval"],
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        superseded_by="FiQA2018-Fa.v2",
    )


class FiQA2018FaV2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018-Fa.v2",
        description="FiQA2018-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/fiqa-fa-v2",
        dataset={
            "path": "MCINext/fiqa-fa-v2",
            "revision": "3443316360bb14c1ef68f609d6fe8f89b117977d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        category="t2t",
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
        superseded_by="QuoraRetrieval-Fa.v2",
    )


class QuoraRetrievalFaV2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="QuoraRetrieval-Fa.v2",
        description="QuoraRetrieval-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/quora-fa-v2",
        dataset={
            "path": "MCINext/quora-fa-v2",
            "revision": "4541104f8c0e1f3932838c99197fc4ce47843267",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        superseded_by="SCIDOCS-Fa.v2",
    )


class SCIDOCSFaV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-Fa.v2",
        description="SCIDOCS-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/scidocs-fa-v2",
        dataset={
            "path": "MCINext/scidocs-fa-v2",
            "revision": "930aea79c05ad5c889399c41132229c1047c8c93",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        superseded_by="SciFact-Fa.v2",
    )


class SciFactFaV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact-Fa.v2",
        description="SciFact-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/scifact-fa-v2",
        dataset={
            "path": "MCINext/scifact-fa-v2",
            "revision": "75937d5e71adf8040e511fab41e06bc88825a52b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        superseded_by="TRECCOVID-Fa.v2",
    )


class TRECCOVIDFaV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID-Fa.v2",
        description="TRECCOVID-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/trec-covid-fa-v2",
        dataset={
            "path": "MCINext/trec-covid-fa-v2",
            "revision": "2290ff195683a996a0d3e17455ce6149d4cc7b58",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
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
        category="t2t",
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
        superseded_by="Touche2020-Fa.v2",
    )


class Touche2020FaV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020-Fa.v2",
        description="Touche2020-Fa.v2",
        reference="https://huggingface.co/datasets/MCINext/touche2020-fa-v2",
        dataset={
            "path": "MCINext/webis-touche2020-v3-fa",
            "revision": "b94a6fb6715c0aa4154c898ad2cebddd6f70c42c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2025-07-01", "2025-07-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["Touche2020"],
    )


class HotpotQAFaHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA-FaHardNegatives",
        description="HotpotQA-FaHardNegatives",
        reference="https://huggingface.co/datasets/MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2",
        dataset={
            "path": "MCINext/HotpotQA_FA_test_top_250_only_w_correct-v2",
            "revision": "42a3220e6adc48e678a6f4bbe49f226ca7d0ed83",
        },
        type="Retrieval",
        category="t2t",
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


class MSMARCOFaHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-FaHardNegatives",
        description="MSMARCO-FaHardNegatives",
        reference="https://huggingface.co/datasets/MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2",
        dataset={
            "path": "MCINext/MSMARCO_FA_test_top_250_only_w_correct-v2",
            "revision": "1c67fef70e75a14e416a5da6e38d0b913f9fcc62",
        },
        type="Retrieval",
        category="t2t",
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
        adapted_from=["MSMARCO"],
    )


class NQFaHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-FaHardNegatives",
        description="NQ-FaHardNegatives",
        reference="https://huggingface.co/datasets/MCINext/NQ_FA_test_top_250_only_w_correct-v2",
        dataset={
            "path": "MCINext/NQ_FA_test_top_250_only_w_correct-v2",
            "revision": "69bb4efac89a25da5e70d15793c4ec1498dbe06f",
        },
        type="Retrieval",
        category="t2t",
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


class FEVERFaHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FEVER-FaHardNegatives",
        description="FEVER-FaHardNegatives",
        reference="https://huggingface.co/datasets/MCINext/FEVER_FA_test_top_250_only_w_correct-v2",
        dataset={
            "path": "MCINext/FEVER_FA_test_top_250_only_w_correct-v2",
            "revision": "6eeb028da2a2c8f888e633a878310bf14ce33e09",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Claim verification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["FEVER"],
    )
