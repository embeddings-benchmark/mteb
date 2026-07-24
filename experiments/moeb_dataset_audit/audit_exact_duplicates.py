from collections import defaultdict

import mteb


def normalize_text(text: str) -> str:
    """Normalize text before checking exact duplicates."""
    return " ".join(text.lower().split())


def find_duplicates(items: dict[str, object]) -> dict[str, list[str]]:
    """
    Group IDs whose normalized texts are identical.
    Returns only groups containing at least two IDs.
    """
    text_to_ids: defaultdict[str, list[str]] = defaultdict(list)

    for item_id, value in items.items():
        if isinstance(value, dict):
            text = str(value.get("text", ""))
        else:
            text = str(value)

        normalized = normalize_text(text)
        text_to_ids[normalized].append(item_id)

    return {
        text: ids
        for text, ids in text_to_ids.items()
        if text and len(ids) > 1
    }


def main() -> None:
    task = mteb.get_task("NanoArguAnaRetrieval")
    task.load_data()

    split = "train"
    queries = task.queries[split]
    corpus = task.corpus[split]

    query_duplicates = find_duplicates(queries)
    corpus_duplicates = find_duplicates(corpus)

    print("=== Dataset size ===")
    print("Queries:", len(queries))
    print("Corpus documents:", len(corpus))

    print("\n=== Exact duplicate summary ===")
    print("Duplicate query groups:", len(query_duplicates))
    print("Duplicate corpus groups:", len(corpus_duplicates))

    print("\n=== Example duplicate queries ===")
    for text, ids in list(query_duplicates.items())[:3]:
        print("IDs:", ids)
        print("Text:", text[:300])
        print()

    print("\n=== Example duplicate corpus documents ===")
    for text, ids in list(corpus_duplicates.items())[:3]:
        print("IDs:", ids)
        print("Text:", text[:300])
        print()


if __name__ == "__main__":
    main()