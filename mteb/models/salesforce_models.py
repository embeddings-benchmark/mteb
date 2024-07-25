from functools import partial

import torch

from mteb.model_meta import ModelMeta

from .instructions import task_to_instruction


def sfr_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def sfr_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use SFR_Embedding_2_R."
        )

    class SFRWrapper(GritLM):
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
                kwargs["instruction"] = sfr_instruction(instruction)
            return super().encode(*args, **kwargs)

        def encode_corpus(self, *args, **kwargs):
            kwargs["is_query"] = False
            return super().encode_corpus(*args, **kwargs)

    return SFRWrapper(**kwargs)


SFR_Embedding_2_R = ModelMeta(
    loader=partial(
        sfr_loader,
        model_name_or_path="Salesforce/SFR-Embedding-2_R",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Salesforce/SFR-Embedding-2_R
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_source=True,
    revision="91762139d94ed4371a9fa31db5551272e0b83818",
    release_date="2024-06-14",  # initial commit of hf model.
)

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(SFR_Embedding_2_R.name, SFR_Embedding_2_R.revision)
    emb = mdl.encode(["Hello, world!"])
