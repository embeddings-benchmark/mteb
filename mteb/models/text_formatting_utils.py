from __future__ import annotations


def corpus_to_texts(
    corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
    sep: str = "\n",
) -> list[str]:
    if isinstance(corpus, dict):
        return [
            (corpus["title"][i] + sep + corpus["text"][i]).strip()  # type: ignore
            if "title" in corpus
            else corpus["text"][i].strip()  # type: ignore
            for i in range(len(corpus["text"]))  # type: ignore
        ]
    else:
        if isinstance(corpus[0], str):
            return corpus
        return [
            (doc["title"] + sep + doc["text"]).strip()
            if "title" in doc
            else doc["text"].strip()
            for doc in corpus
        ]
