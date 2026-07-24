from collections import defaultdict

import mteb


def normalize_text(text: str) -> str:
    """Ignore capitalization and whitespace differences."""
    return " ".join(text.lower().split())


def extract_text(value: object) -> str:
    """Extract text from either a document dictionary or a plain string."""
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def get_relevant_doc_ids(labels: object) -> set[str]:
    """Convert MTEB relevance labels into a set of relevant document IDs."""
    if isinstance(labels, dict):
        return set(labels.keys())

    if isinstance(labels, (list, tuple, set)):
        return set(labels)

    raise TypeError(f"Unsupported relevance-label type: {type(labels)}")


def main() -> None:
    task = mteb.get_task("NanoArguAnaRetrieval")
    task.load_data()

    split = "train"
    corpus = task.corpus[split]
    queries = task.queries[split]
    relevant_docs = task.relevant_docs[split]

    # Step 1: Group document IDs that have identical normalized text.
    text_to_doc_ids: defaultdict[str, list[str]] = defaultdict(list)

    for doc_id, document in corpus.items():
        normalized_text = normalize_text(extract_text(document))

        if normalized_text:
            text_to_doc_ids[normalized_text].append(doc_id)

    duplicate_groups = {
        text: doc_ids
        for text, doc_ids in text_to_doc_ids.items()
        if len(doc_ids) > 1
    }

    # Step 2: For each query, check whether only some IDs in an
    # identical-document group are marked relevant.
    inconsistencies: list[dict[str, object]] = []

    for query_id, labels in relevant_docs.items():
        relevant_ids = get_relevant_doc_ids(labels)

        for text, duplicate_doc_ids in duplicate_groups.items():
            duplicate_id_set = set(duplicate_doc_ids)
            marked_relevant = duplicate_id_set & relevant_ids

            if marked_relevant and marked_relevant != duplicate_id_set:
                inconsistencies.append(
                    {
                        "query_id": query_id,
                        "query_text": queries[query_id],
                        "duplicate_doc_ids": duplicate_doc_ids,
                        "marked_relevant": sorted(marked_relevant),
                        "not_marked_relevant": sorted(
                            duplicate_id_set - marked_relevant
                        ),
                        "document_text": text,
                    }
                )

    total_documents_in_duplicate_groups = sum(
        len(doc_ids) for doc_ids in duplicate_groups.values()
    )

    redundant_document_copies = sum(
        len(doc_ids) - 1 for doc_ids in duplicate_groups.values()
    )

    print("=== Duplicate summary ===")
    print("Corpus documents:", len(corpus))
    print("Duplicate text groups:", len(duplicate_groups))
    print(
        "Documents in duplicate groups:",
        total_documents_in_duplicate_groups,
    )
    print("Redundant document copies:", redundant_document_copies)

    print("\n=== Potential qrel inconsistencies ===")
    print("Affected query/group cases:", len(inconsistencies))

    for case in inconsistencies[:10]:
        print("\n--- Potential inconsistency ---")
        print("Query ID:", case["query_id"])
        print("Query:", str(case["query_text"])[:500])
        print("All duplicate document IDs:", case["duplicate_doc_ids"])
        print("Marked relevant:", case["marked_relevant"])
        print("Not marked relevant:", case["not_marked_relevant"])
        print("Duplicate document text:", str(case["document_text"])[:500])


if __name__ == "__main__":
    main()