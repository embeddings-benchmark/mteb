from mteb.abstasks.spatial_arrangement import AbsTaskSpatialArrangement
from mteb.abstasks.task_metadata import TaskMetadata


class StakeholderGrounding(AbsTaskSpatialArrangement):
    metadata = TaskMetadata(
        name="StakeholderGrounding",
        description="Danish technology practitioners' statements on challenges with Responsible AI, spatially arranged by domain experts (Spatial Arrangement Method) so that canvas proximity reflects perceived similarity. Each sample is one arrangement of 20 statements by one expert. Models are scored by triplet-based Krippendorff's alpha AUC against each expert layout.",
        reference="https://arxiv.org/abs/2605.27168",
        dataset={
            "path": "ryzzlestrizzle/stakeholder-grounding",
            "revision": "836f23984a2761a3a518c6c2582736779935f457",
        },
        type="SpatialArrangement",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"rai": ["dan-Latn"]},
        main_score="alpha_auc",
        date=("2025-06-03", "2025-06-03"),  # TODO: survey and workshop collection dates
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{rystrom2026grounding,
  archiveprefix = {arXiv},
  author = {Jonathan Rystrøm and Sofie Burgos-Thorsen and Zihao Fu and Johan Irving Søltoft and Kenneth C. Enevoldsen and Chris Russell},
  eprint = {2605.27168},
  primaryclass = {cs.CL},
  title = {Grounding Text Embeddings in Stakeholder Associations},
  year = {2026},
}
""",
    )
