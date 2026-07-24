import mteb


def main() -> None:
    task = mteb.get_task("NanoArguAnaRetrieval")
    task.load_data()

    corpus = task.corpus["train"]

    doc_id_1 = "test-economy-epiasghbf-con01b"
    doc_id_2 = "test-society-epiasghbf-con01b"

    doc_1 = corpus[doc_id_1]
    doc_2 = corpus[doc_id_2]

    print("=== Document 1 ===")
    print("ID:", doc_id_1)
    print(doc_1)

    print("\n=== Document 2 ===")
    print("ID:", doc_id_2)
    print(doc_2)

    text_1 = doc_1["text"]
    text_2 = doc_2["text"]

    print("\n=== Comparison ===")
    print("Raw text equal:", text_1 == text_2)
    print(
        "Normalized text equal:",
        " ".join(text_1.lower().split())
        == " ".join(text_2.lower().split()),
    )

    query_id = "test-economy-epiasghbf-con01a"
    relevant = task.relevant_docs["train"][query_id]

    print("\n=== Qrel check ===")
    print("Relevant docs:", relevant)
    print("Document 1 marked relevant:", doc_id_1 in relevant)
    print("Document 2 marked relevant:", doc_id_2 in relevant)


if __name__ == "__main__":
    main()