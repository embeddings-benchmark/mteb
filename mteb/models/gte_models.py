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
    import mteb

    mdl = mteb.get_model(gte_Qwen2_7B_instruct.name, gte_Qwen2_7B_instruct.revision)
    emb = mdl.encode(["Hello, world!"])
