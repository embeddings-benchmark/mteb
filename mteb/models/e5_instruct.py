from functools import partial

import torch

from mteb.model_meta import ModelMeta

from .e5_models import E5_PAPER_RELEASE_DATE, XLMR_LANGUAGES
from .instructions import task_to_instruction

MISTRAL_LANGUAGES = ["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"]


def e5_instruction(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: "


def e5_loader(**kwargs):
    try:
        from gritlm import GritLM
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use E5 Instruct models."
        )

    class E5InstructWrapper(GritLM):
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
                kwargs["instruction"] = e5_instruction(instruction)
            return super().encode(*args, **kwargs)

        def encode_corpus(self, *args, **kwargs):
            kwargs["is_query"] = False
            return super().encode_corpus(*args, **kwargs)

    return E5InstructWrapper(**kwargs)


e5_instruct = ModelMeta(
    loader=partial(
        e5_loader,
        model_name_or_path="intfloat/multilingual-e5-large-instruct",
        attn="cccc",
        pooling_method="mean",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
    ),
    name="intfloat/multilingual-e5-large-instruct",
    languages=XLMR_LANGUAGES,
    open_source=True,
    revision="baa7be480a7de1539afce709c8f13f833a510e0a",
    release_date=E5_PAPER_RELEASE_DATE,
)

e5_mistral = ModelMeta(
    loader=partial(
        e5_loader,
        model_name_or_path="intfloat/e5-mistral-7b-instruct",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct#transformers
        normalized=True,
    ),
    name="intfloat/e5-mistral-7b-instruct",
    languages=MISTRAL_LANGUAGES,
    open_source=True,
    revision="07163b72af1488142a360786df853f237b1a3ca1",
    release_date=E5_PAPER_RELEASE_DATE,
)
