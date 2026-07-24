import mteb


def describe(name: str, value: object) -> None:
    print(f"\n{name}:")
    print(f"  type = {type(value)}")

    if value is None:
        print("  value = None")
        return

    if isinstance(value, dict):
        print(f"  keys = {list(value.keys())}")

    try:
        print(f"  length = {len(value)}")  # type: ignore[arg-type]
    except TypeError:
        pass


def main() -> None:
    task = mteb.get_task("NanoArguAnaRetrieval")
    task.load_data()

    print("task",task)
    print("Task class:", type(task))
    print("Data loaded:", task.data_loaded)
    print("Instance attributes:", list(vars(task).keys()))

    for attribute_name in [
        "dataset",
        "corpus",
        "queries",
        "relevant_docs",
        "qrels",
    ]:
        describe(
            attribute_name,
            getattr(task, attribute_name, None),
        )
    
    split = "train"

    corpus = task.corpus[split]
    queries = task.queries[split]
    relevant_docs = task.relevant_docs[split]

    print("\nCorpus count:", len(corpus))
    print("Query count:", len(queries))
    print("Relevant-doc count:", len(relevant_docs))

    # 取第一个 query
    query_id, query_text = next(iter(queries.items()))

    print("\n=== Example query ===")
    print("Query ID:", query_id)
    print("Query:", query_text)

    print("\n=== Relevant document IDs ===")
    print(relevant_docs[query_id])

    relevant_document_id = next(iter(relevant_docs[query_id]))
    relevant_document = corpus[relevant_document_id]

    print("\n=== Relevant document ===")
    print("Document ID:", relevant_document_id)
    print("Document:", relevant_document)


if __name__ == "__main__":
    main()
