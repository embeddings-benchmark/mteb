from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta

from .instructions import task_to_instruction


def gte_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def gte_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use gte-Qwen2-7B-instruct."
        )

    class GTEWrapper(GritLM):
        def encode(self, *args, **kwargs):
            if "prompt_name" in kwargs:
                if "instruction" in kwargs:
                    raise ValueError(
                        "Cannot specify both `prompt_name` and `instruction`."
                    )
                instruction = task_to_instruction(
                    kwargs.pop("prompt_name"), kwargs.pop("is_query", True)
                )
            else:
                instruction = kwargs.pop("instruction", "")
            if instruction:
                kwargs["instruction"] = gte_instruction(instruction)
            return super().encode(*args, **kwargs)

        def encode_corpus(self, *args, **kwargs):
            kwargs["is_query"] = False
            return super().encode_corpus(*args, **kwargs)

    return GTEWrapper(**kwargs)


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(
        gte_loader,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_source=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
)


if __name__ == "__main__":
    # Verify it reproduces https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
    )
    # Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]
    # Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    # In case you want to reduce the maximum length:
    model.max_seq_length = 8192
    queries = ["how much protein should a female eat", "summit define"]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)
    scores = (query_embeddings @ document_embeddings.T) * 100
    print(scores.tolist())
    # [[70.39706420898438, 3.4318461418151855], [4.516170978546143, 81.91815948486328]]

    import mteb

    model_mteb = mteb.get_model(
        "Alibaba-NLP/gte-Qwen2-7B-instruct"
    )  # gte_Qwen2_7B_instruct.name, gte_Qwen2_7B_instruct.revision)
    # Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
    # Created GritLM: torch.float32 dtype, lasttoken pool, embedding mode, cccc attn
    # Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    # ----------Using 8 data-parallel GPUs----------
    query_embeddings_mteb = model_mteb.encode(
        queries,
        instruction="Given a web search query, retrieve relevant passages that answer the query",
    )
    document_embeddings_mteb = model_mteb.encode_corpus(documents)
    scores_mteb = (query_embeddings_mteb @ document_embeddings_mteb.T) * 100
    print(scores_mteb.tolist())
    # [[70.39706420898438, 3.4318461418151855], [4.516170978546143, 81.91815948486328]]
