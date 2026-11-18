from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://arxiv.org/abs/1912.03098"
_BIBTEX = r"""
@inproceedings{pont-tuset2020localized,
  author = {Pont-Tuset, Jordi and Uijlings, Jasper and Changpinyo, Soravit and Soricut, Radu and Ferrari, Vittorio},
  booktitle = {Computer Vision -- ECCV 2020},
  doi = {10.1007/978-3-030-58558-7_38},
  pages = {647--664},
  title = {Connecting Vision and Language with Localized Narratives},
  year = {2020},
}
"""
_DESCRIPTION = (
    "Localized Narratives asked annotators to describe an image out loud while "
    "looking at it, producing unscripted spoken narrations. This task uses the "
    "Open Images test subset (the COCO and Flickr30k subsets overlap with existing "
    "spoken-caption tasks): 2,000 images sampled with a fixed seed, one narration "
    "per image, with the original recordings resampled to 16 kHz and not trimmed. "
)


class LocalizedNarrativesA2IRetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata = TaskMetadata(
        name="LocalizedNarrativesA2IRetrieval",
        description=_DESCRIPTION
        + "Queries are spoken narrations and the corpus contains images; the goal is "
        "to retrieve the image the speaker was describing.",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/LocalizedNarrativesA2I",
            "revision": "896f4a9ebe753dd01e76bf0c5d2748f3767346a1",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the image described by this spoken narration."},
    )


class LocalizedNarrativesI2ARetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata = TaskMetadata(
        name="LocalizedNarrativesI2ARetrieval",
        description=_DESCRIPTION
        + "Queries are images and the corpus contains spoken narrations; the goal is "
        "to retrieve the narration recorded for each image.",
        reference=_REFERENCE,
        dataset={
            "path": "mteb/LocalizedNarrativesI2A",
            "revision": "74390f059837a7aa1df96697a6b009eabb4d6b51",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Scene", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the spoken narration that describes this image."},
    )
