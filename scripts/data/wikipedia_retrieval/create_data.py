from datasets import load_dataset, Dataset, DatasetDict, Features, Value

languages = ["de"]

def apply_query_id(example, queries_dict):
    title = example["title"]
    corpus_id = example["_id"]
    score = example["score"]
    query_id = queries_dict[title]
    return {"query-id": query_id, "corpus-id": corpus_id, "score": score}

for lang in languages:
    ds_queries = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-queries", split="train")
    ds_corpus = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-1.5k-articles", split="train")

    queries = ds_queries.map(lambda x: {"_id": "q" + x["_id"], "text": x["query"]}, remove_columns=ds_queries.column_names)
    queries_dict = {row["title"]: "q" + row["_id"] for row in ds_queries}
    corpus = ds_corpus.map(lambda x: {"_id": x["_id"], "title": x["title"], "text": x["text"]}, remove_columns=ds_corpus.column_names)
    qrels = ds_corpus.map(lambda x: apply_query_id(x, queries_dict=queries_dict), remove_columns=ds_corpus.column_names)
    # breakpoint()

    corpus_features = Features({"_id": Value("string"), "title": Value("string"), "text": Value("string")})
    queries_features = Features({"_id": Value("string"), "text": Value("string")})
    qrels_features = Features({"query-id": Value("string"), "corpus-id": Value("string"), "score": Value("float32")})

    # corpus_dataset = Dataset.from_dict(corpus, features=corpus_features, split="test")
    # queries_dataset = Dataset.from_dict(queries, features=queries_features, split="test")
    # qrels_dataset = Dataset.from_dict(qrels, features=qrels_features, split="test")

    # ds_dict = DatasetDict({"corpus": corpus_dataset, "queries": queries_dataset, "qrels": qrels_dataset})
    # ds_dict.save_to_disk(f"dataset/wikipedia-2023-11-retrieval-{lang}")

    corpus = corpus.cast(corpus_features)
    queries = queries.cast(queries_features)
    qrels = qrels.cast(qrels_features)

    ds_dict = DatasetDict({"corpus": corpus, "queries": queries, "qrels": qrels})
    # ds_dict.save_to_disk(f"datasets/wikipedia-2023-11-retrieval-{lang}")
    # ds_dict.push_to_hub(f"ellamind/wikipedia-2023-11-retrieval-{lang}")

    corpus.push_to_hub(f"ellamind/wikipedia-2023-11-retrieval-{lang}", config_name="corpus", split="test")
    queries.push_to_hub(f"ellamind/wikipedia-2023-11-retrieval-{lang}", config_name="queries", split="test")
    qrels.push_to_hub(f"ellamind/wikipedia-2023-11-retrieval-{lang}", config_name="qrels", split="test")

    # qrels_datadict = DatasetDict({"test": qrels_dataset})

    # corpus_datadict.save_to_disk(f"scripts/data/wikipedia-retrieval-{lang}/corpus_{lang}")
    # qrels_datadict.save_to_disk(f"scripts/data/germanquad/qrels_{lang}")