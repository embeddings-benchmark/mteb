import hashlib
from collections import Counter

from PIL import Image

from mteb.types import TopRankedDocumentsType
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    RelevantDocsStatistics,
    ScoreStatistics,
    TextStatistics,
    TopRankedStatistics,
)


def calculate_text_statistics(texts: list[str]) -> TextStatistics:
    """Calculate descriptive statistics for a list of texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        TextStatistics: A dictionary containing the descriptive statistics.
    """
    lengths = [len(text) for text in texts]
    unique_texts = len(set(texts))

    return TextStatistics(
        total_text_length=sum(lengths),
        min_text_length=min(lengths),
        average_text_length=sum(lengths) / len(lengths),
        max_text_length=max(lengths),
        unique_texts=unique_texts,
    )


def calculate_image_statistics(images: list[Image.Image]) -> ImageStatistics:
    """Calculate descriptive statistics for a list of images.

    Args:
        images: List of images to analyze. Each image should have a `size` attribute that returns a tuple (width, height).

    Returns:
        ImageStatistics: A dictionary containing the descriptive statistics.
    """
    img_widths, img_heights = [], []
    seen_hashes: set[str] = set()

    for img in images:
        width, height = img.size  # type: ignore
        img_heights.append(height)
        img_widths.append(width)

        img_bytes = img.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        seen_hashes.add(img_hash)

    return ImageStatistics(
        min_image_width=min(img_widths),
        average_image_width=sum(img_widths) / len(img_widths),
        max_image_width=max(img_widths),
        min_image_height=min(img_heights),
        average_image_height=sum(img_heights) / len(img_heights),
        max_image_height=max(img_heights),
        # some image types (PngImageFile) may be unhashable
        unique_images=len(seen_hashes),
    )


def calculate_label_statistics(labels: list[int | list[int]]) -> LabelStatistics:
    """Calculate descriptive statistics for a list of labels.

    Args:
        labels: List of labels, where each label can be an integer or a list of integers (for multilabel classification).

    Returns:
        LabelStatistics: A dictionary containing the descriptive statistics.

    """
    if not isinstance(labels[0], list):
        label_len = [1] * len(labels)
        total_label_len = len(labels)
        total_labels = labels
    elif isinstance(labels[0], list):
        # multilabel classification
        label_len = [len(l) for l in labels]
        total_label_len = sum(label_len)
        total_labels = []
        for l in labels:
            total_labels.extend(l if len(l) > 0 else [None])
    else:
        raise ValueError(
            "Labels must be a list of integers or a list of lists of integers."
        )

    label_count = Counter(total_labels)
    return LabelStatistics(
        min_labels_per_text=min(label_len),
        average_label_per_text=total_label_len / len(labels),
        max_labels_per_text=max(label_len),
        unique_labels=len(label_count),
        labels={
            str(label): {
                "count": value,
            }
            for label, value in label_count.items()
        },
    )


def calculate_score_statistics(scores: list[int | float]) -> ScoreStatistics:
    """Calculate descriptive statistics for a list of scores.

    Args:
        scores: List of scores to analyze.

    Returns:
        ScoreStatistics: A dictionary containing the descriptive statistics.
    """
    return ScoreStatistics(
        min_score=min(scores),
        avg_score=sum(scores) / len(scores),
        max_score=max(scores),
    )


def calculate_top_ranked_statistics(
    top_ranked: TopRankedDocumentsType, num_queries: int
) -> TopRankedStatistics:
    """Calculate statistics for top-ranked items.

    Args:
        top_ranked: List of lists, where each inner list contains IDs of top-ranked items.
        num_queries: Total number of queries.

    Returns:
        dict: A dictionary with the count of top-ranked items per ID.
    """
    return TopRankedStatistics(
        num_top_ranked=sum(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
        min_top_ranked_per_query=min(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
        average_top_ranked_per_query=(
            sum(len(docs) for docs in top_ranked.values() if docs is not None)
            / num_queries
        ),
        max_top_ranked_per_query=max(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
    )


def calculate_relevant_docs_statistics(
    relevant_docs: dict[str, dict[str, float]],
) -> RelevantDocsStatistics:
    qrels_lengths = [len(relevant_docs[qid]) for qid in relevant_docs]
    unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})
    # number of qrels that are not 0
    num_qrels_non_zero = sum(
        sum(1 for doc_id in docs if docs[doc_id] != 0)
        for docs in relevant_docs.values()
    )
    qrels_per_doc = num_qrels_non_zero / len(relevant_docs)

    return RelevantDocsStatistics(
        num_relevant_docs=num_qrels_non_zero,
        min_relevant_docs_per_query=min(qrels_lengths),
        average_relevant_docs_per_query=qrels_per_doc,
        max_relevant_docs_per_query=max(qrels_lengths),
        unique_relevant_docs=unique_qrels,
    )
