from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_climate_fever_metadata = dict(
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("2001-01-01", "2020-12-31"),  # launch of wiki -> paper publication
    domains=["Encyclopaedic", "Written"],
    task_subtypes=["Claim verification"],
    license="cc-by-sa-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@misc{diggelmann2021climatefever,
  archiveprefix = {arXiv},
  author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
  eprint = {2012.00614},
  primaryclass = {cs.CL},
  title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
  year = {2021},
}
""",
)


class ClimateFEVER(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER",
        description=(
            "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims "
            "(queries) regarding climate-change. The underlying corpus is the same as FEVER."
        ),
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={
            "path": "mteb/climate-fever",
            "revision": "47f2ac6acb640fc46020b02a5b59fdda04d39380",
        },
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
        **_climate_fever_metadata,
    )


class ClimateFEVERRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVER.v2",
        description=(
            "CLIMATE-FEVER is a dataset following the FEVER methodology, containing 1,535 real-world climate change claims. "
            "This updated version addresses corpus mismatches and qrel inconsistencies in MTEB, restoring labels while refining corpus-query alignment for better accuracy."
        ),
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={
            "path": "mteb/climate-fever-v2",
            "revision": "e438c9586767800aeb10dbe8a245c41dbea4e5f4",
        },
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
        adapted_from=["ClimateFEVER"],
        **_climate_fever_metadata,
    )


class ClimateFEVERHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVERHardNegatives",
        description=(
            "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={
            "path": "mteb/ClimateFEVER_test_top_250_only_w_correct-v2",
            "revision": "3a309e201f3c2c4b13bd4a367a8f37eee2ec1d21",
        },
        adapted_from=["ClimateFEVER"],
        superseded_by="ClimateFEVERHardNegatives.v2",
        **_climate_fever_metadata,
    )


class ClimateFEVERHardNegativesV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVERHardNegatives.v2",
        description=(
            "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. "
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={
            "path": "mteb/ClimateFEVER_test_top_250_only_w_correct-v2",
            "revision": "3a309e201f3c2c4b13bd4a367a8f37eee2ec1d21",
        },
        adapted_from=["ClimateFEVER"],
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
        **_climate_fever_metadata,
    )
