import logging
from functools import partial

from mteb.model_meta import ModelMeta

from .instructions import task_to_instruction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def gritlm_instruction(instruction):
    return (
        "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    )


def gritlm_loader(**kwargs):
    try:
        from gritlm import GritLM

        class GritLMWrapper(GritLM):
            def encode(self, *args, **kwargs):
                if "prompt_name" in kwargs:
                    instruction = gritlm_instruction(
                        task_to_instruction(
                            kwargs.pop("prompt_name"), kwargs.get("is_query", True)
                        )
                    )
                else:
                    instruction = gritlm_instruction("")
                kwargs["instruction"] = instruction
                return super().encode(*args, **kwargs)

            def encode_corpus(self, *args, **kwargs):
                kwargs["is_query"] = False
                return super().encode_corpus(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "GritLM is not installed. Please install it with `pip install gritlm`."
        )
    kwargs.pop("device", None)  # GritLM does automatic device placement
    return GritLMWrapper(**kwargs)


gritlm7b = ModelMeta(
    loader=partial(
        gritlm_loader,
        model_name_or_path="GritLM/GritLM-7B",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    open_source=True,
    revision="13f00a0e36500c80ce12870ea513846a066004af",
    release_date="2024-02-15",
)
gritlm8x7b = ModelMeta(
    loader=partial(
        gritlm_loader,
        model_name_or_path="GritLM/GritLM-8x7B",
        mode="embedding",
        torch_dtype="auto",
    ),
    name="GritLM/GritLM-8x7B",
    languages=["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"],
    open_source=True,
    revision="7f089b13e3345510281733ca1e6ff871b5b4bc76",
    release_date="2024-02-15",
)
