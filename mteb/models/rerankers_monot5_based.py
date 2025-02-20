from __future__ import annotations

import logging
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper, _loader

logger = logging.getLogger(__name__)


# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    "castorini/monot5-small-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-small-msmarco-100k": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
}


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class MonoT5Reranker(RerankerWrapper):
    name: str = "MonoT5"
    prompt_template: str = "Query: {query} Document: {text} Relevant:"

    def __init__(
        self,
        model_name_or_path="castorini/monot5-base-msmarco-10k",
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, **model_args
        )
        logger.info(f"Using model {model_name_or_path}")

        if "torch_compile" in kwargs and kwargs["torch_compile"]:
            self.torch_compile = kwargs["torch_compile"]
            self.model = torch.compile(self.model)
        else:
            self.torch_compile = False

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path,
            self.tokenizer,
            kwargs["token_false"] if "token_false" in kwargs else None,
            kwargs["token_true"] if "token_true" in kwargs else None,
        )
        logger.info(f"Using max_length of {self.tokenizer.model_max_length}")
        logger.info(f"Using token_false_id of {self.token_false_id}")
        logger.info(f"Using token_true_id of {self.token_true_id}")
        self.max_length = min(
            2048, self.tokenizer.model_max_length
        )  # sometimes it's a v large number/max int
        logger.info(f"Using max_length of {self.max_length}")

        self.model.eval()

    def get_prediction_tokens(
        self, model_name_or_path, tokenizer, token_false=None, token_true=None
    ):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                raise Exception(
                    f"We don't know the indexes for the non-relevant/relevant tokens for\
                        the checkpoint {model_name_or_path} and you did not provide any."
                )
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        prompts = [
            self.prompt_template.format(query=query, text=text)
            for (query, text) in zip(queries, passages)
        ]

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            pad_to_multiple_of=(8 if self.torch_compile else None),
        ).to(self.device)
        output = self.model.generate(
            **tokens,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        batch_scores = output.scores[0]
        batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()


class LlamaReranker(RerankerWrapper):
    name: str = "LLAMA-Based"

    def __init__(
        self, model_name_or_path: str, is_classification: bool = False, **kwargs
    ):
        if "torch_compile" in kwargs:
            del kwargs["torch_compile"]
        super().__init__(model_name_or_path, **kwargs)

        if "chat" in model_name_or_path:
            self.template = """<s>[INST] <<SYS>>
You are an expert at finding information. Determine if the following document is relevant to the query (true/false).
<</SYS>>Query: {query}
Document: {text}
Relevant: [/INST]"""
        else:
            self.template = """Determine if the following document is relevant to the query (true/false).

Query: {query}
Document: {text}
Relevant: """

        self.query_instruct_template = "{query} {instruction}"
        logger.info(f"Using query_instruct_template of {self.query_instruct_template}")
        self.is_classification = is_classification

        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        logger.info(self.template)
        logger.info(model_name_or_path)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_args
        )
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id = self.tokenizer.get_vocab()["true"]
        self.max_length = min(2048, self.tokenizer.model_max_length)
        logger.info(f"Using max_length of {self.max_length}")
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            logger.info(f"Using {self.gpu_count} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            # logger.info(f"Adding instructions to LLAMA queries")
            queries = [
                self.query_instruct_template.format(instruction=i, query=q).strip()
                for i, q in zip(instructions, queries)
            ]

        prompts = [
            self.template.format(query=query, text=text)
            for (query, text) in zip(queries, passages)
        ]
        assert "{query}" not in prompts[0], "Query not replaced"
        assert "{text}" not in prompts[0], "Text not replaced"
        assert "{instruction}" not in prompts[0], "Instruction not replaced"

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            pad_to_multiple_of=None,
        ).to(self.device)
        if "token_type_ids" in tokens:
            del tokens["token_type_ids"]
        if not self.is_classification:
            batch_scores = self.model(**tokens).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        else:
            batch_scores = self.model(**tokens).logits
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        return scores


class MistralReranker(LlamaReranker):
    name: str = "Mistral"

    def __init__(self, model_name_or_path: str, **kwargs):
        # use the base class for everything except template
        super().__init__(model_name_or_path, **kwargs)
        self.template = """<s>[INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false).
Query: {query}
Document: {text}
Relevant (either "true" or "false"): [/INST]"""
        self.max_length = min(2048, self.tokenizer.model_max_length)
        logger.info(f"Using max_length of {self.max_length}")
        logger.info(f"Using template of {self.template}")


class FollowIRReranker(LlamaReranker):
    name: str = "FollowIR"

    def __init__(self, model_name_or_path: str, **kwargs):
        # use the base class for everything except template
        super().__init__(model_name_or_path, **kwargs)
        self.template = """<s> [INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.

Query: {query}
Document: {text}
Relevant (only output one word, either "true" or "false"): [/INST] """
        self.max_length = min(2048, self.tokenizer.model_max_length)
        logger.info(f"Using template of {self.template}")


class FLANT5Reranker(MonoT5Reranker):
    name: str = "FLAN-T5"
    prompt_template: str = """Is the following passage relevant to the query?
Query: {query}
Passage: {text}"""

    def get_prediction_tokens(self, *args, **kwargs):
        yes_token_id, *_ = self.tokenizer.encode("yes")
        no_token_id, *_ = self.tokenizer.encode("no")
        return no_token_id, yes_token_id


monot5_small = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="castorini/monot5-small-msmarco-10k",
        fp_options="float16",
    ),
    name="castorini/monot5-small-msmarco-10k",
    languages=["eng_Latn"],
    open_weights=True,
    revision="77f8e3f7b1eb1afe353aa21a7c3a2fc8feca702e",
    release_date="2022-03-28",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

monot5_base = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="castorini/monot5-base-msmarco-10k",
        fp_options="float16",
    ),
    name="castorini/monot5-base-msmarco-10k",
    languages=["eng_Latn"],
    open_weights=True,
    revision="f15657ab3d2a5dd0b9a30c8c0b6a0a73c9cb5884",
    release_date="2022-03-28",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

monot5_large = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="castorini/monot5-large-msmarco-10k",
        fp_options="float16",
    ),
    name="castorini/monot5-large-msmarco-10k",
    languages=["eng_Latn"],
    open_weights=True,
    revision="48cfad1d8dd587670393f27ee8ec41fde63e3d98",
    release_date="2022-03-28",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

monot5_3b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="castorini/monot5-3b-msmarco-10k",
        fp_options="float16",
    ),
    name="castorini/monot5-3b-msmarco-10k",
    languages=["eng_Latn"],
    open_weights=True,
    revision="bc0c419a438c81f592f878ce32430a1823f5db6c",
    release_date="2022-03-28",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

flant5_base = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=FLANT5Reranker,
        model_name_or_path="google/flan-t5-base",
        fp_options="float16",
    ),
    name="google/flan-t5-base",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    release_date="2022-10-21",
    training_datasets={
        "svakulenk0/qrecc": ["train"],
        "taskmaster2": ["train"],
        "djaym7/wiki_dialog": ["train"],
        "deepmind/code_contests": ["train"],
        "lambada": ["train"],
        "gsm8k": ["train"],
        "aqua_rat": ["train"],
        "esnli": ["train"],
        "quasc": ["train"],
        "qed": ["train"],
    },
    n_parameters=248_000_000,
    memory_usage_mb=944,
    max_tokens=None,
    embed_dim=768,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=True,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

flant5_large = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=FLANT5Reranker,
        model_name_or_path="google/flan-t5-large",
        fp_options="float16",
    ),
    name="google/flan-t5-large",
    languages=["eng_Latn"],
    open_weights=True,
    revision="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    release_date="2022-10-21",
    training_datasets={
        "svakulenk0/qrecc": ["train"],
        "taskmaster2": ["train"],
        "djaym7/wiki_dialog": ["train"],
        "deepmind/code_contests": ["train"],
        "lambada": ["train"],
        "gsm8k": ["train"],
        "aqua_rat": ["train"],
        "esnli": ["train"],
        "quasc": ["train"],
        "qed": ["train"],
    },
    n_parameters=783_000_000,
    max_tokens=1024,
    memory_usage_mb=2987,
    embed_dim=None,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

flant5_xl = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=FLANT5Reranker,
        model_name_or_path="google/flan-t5-xl",
        fp_options="float16",
    ),
    name="google/flan-t5-xl",
    languages=["eng_Latn"],
    open_weights=True,
    revision="7d6315df2c2fb742f0f5b556879d730926ca9001",
    release_date="2022-10-21",
    training_datasets={
        "svakulenk0/qrecc": ["train"],
        "taskmaster2": ["train"],
        "djaym7/wiki_dialog": ["train"],
        "deepmind/code_contests": ["train"],
        "lambada": ["train"],
        "gsm8k": ["train"],
        "aqua_rat": ["train"],
        "esnli": ["train"],
        "quasc": ["train"],
        "qed": ["train"],
    },
    n_parameters=2_850_000_000,
    memory_usage_mb=10871,
    max_tokens=None,
    embed_dim=2048,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

flant5_xxl = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=FLANT5Reranker,
        model_name_or_path="google/flan-t5-xxl",
        fp_options="float16",
    ),
    name="google/flan-t5-xxl",
    languages=["eng_Latn"],
    open_weights=True,
    revision="ae7c9136adc7555eeccc78cdd960dfd60fb346ce",
    release_date="2022-10-21",
    training_datasets={
        "svakulenk0/qrecc": ["train"],
        "taskmaster2": ["train"],
        "djaym7/wiki_dialog": ["train"],
        "deepmind/code_contests": ["train"],
        "lambada": ["train"],
        "gsm8k": ["train"],
        "aqua_rat": ["train"],
        "esnli": ["train"],
        "quasc": ["train"],
        "qed": ["train"],
    },
    n_parameters=11_300_000_000,
    memory_usage_mb=42980,
    max_tokens=None,
    embed_dim=4096,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)


llama2_7b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=LlamaReranker,
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        fp_options="float16",
    ),
    name="meta-llama/Llama-2-7b-hf",
    languages=["eng_Latn"],
    open_weights=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9",
    release_date="2023-07-18",
    n_parameters=6_740_000_000,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,  # llama2
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

llama2_7b_chat = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=LlamaReranker,
        model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        fp_options="float16",
    ),
    name="meta-llama/Llama-2-7b-chat-hf",
    languages=["eng_Latn"],
    open_weights=True,
    revision="f5db02db724555f92da89c216ac04704f23d4590",
    release_date="2023-07-18",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

mistral_7b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MistralReranker,
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        fp_options="float16",
    ),
    name="mistralai/Mistral-7B-Instruct-v0.2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="3ad372fc79158a2148299e3318516c786aeded6c",
    release_date="2023-12-11",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

followir_7b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=FollowIRReranker,
        model_name_or_path="jhu-clsp/FollowIR-7B",
        fp_options="float16",
    ),
    name="jhu-clsp/FollowIR-7B",
    languages=["eng_Latn"],
    open_weights=True,
    revision="4d25d437e38b510c01852070c0731e8f6e1875d1",
    release_date="2024-04-29",
    training_datasets={"jhu-clsp/FollowIR-train": ["train"]},
    n_parameters=7_240_000_000,
    memory_usage_mb=13813,
    max_tokens=None,
    embed_dim=None,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)


mt5_languages = [
    "afr_Latn",
    "sqi_Latn",
    "amh_Ethi",
    "ara_Arab",
    "hye_Armn",
    "aze_Latn",
    "eus_Latn",
    "bel_Cyrl",
    "ben_Beng",
    "bul_Cyrl",
    "mya_Mymr",
    "cat_Latn",
    "ceb_Latn",
    "nya_Latn",
    "zho_Hans",
    "cos_Latn",
    "ces_Latn",
    "dan_Latn",
    "nld_Latn",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "fil_Latn",
    "fin_Latn",
    "fra_Latn",
    "glg_Latn",
    "kat_Geor",
    "deu_Latn",
    "ell_Grek",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "haw_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hmn_Latn",
    "hun_Latn",
    "isl_Latn",
    "ibo_Latn",
    "ind_Latn",
    "gle_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "jav_Latn",
    "kan_Knda",
    "kaz_Cyrl",
    "khm_Khmr",
    "kor_Hang",
    "kur_Latn",
    "kir_Cyrl",
    "lao_Laoo",
    "lat_Latn",
    "lav_Latn",
    "lit_Latn",
    "ltz_Latn",
    "mkd_Cyrl",
    "mlg_Latn",
    "msa_Latn",
    "mal_Mlym",
    "mlt_Latn",
    "mri_Latn",
    "mar_Deva",
    "mon_Cyrl",
    "nep_Deva",
    "nor_Latn",
    "pus_Arab",
    "fas_Arab",
    "pol_Latn",
    "por_Latn",
    "pan_Guru",
    "ron_Latn",
    "rus_Cyrl",
    "smo_Latn",
    "gla_Latn",
    "srp_Cyrl",
    "sna_Latn",
    "snd_Arab",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "sun_Latn",
    "swa_Latn",
    "swe_Latn",
    "tgk_Cyrl",
    "tam_Taml",
    "tel_Telu",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "uzb_Latn",
    "vie_Latn",
    "cym_Latn",
    "fry_Latn",
    "xho_Latn",
    "yid_Hebr",
    "yor_Latn",
    "zul_Latn",
]

mt5_base_mmarco_v2 = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="unicamp-dl/mt5-base-mmarco-v2",
        fp_options="float16",
    ),
    name="unicamp-dl/mt5-base-mmarco-v2",
    languages=mt5_languages,
    open_weights=True,
    revision="cc0a949b9f21efcaba45c8cabb998ad02ce8d4e7",
    release_date="2022-01-05",
    training_datasets={"msmarco": ["train"]},
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license="mit",
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)

mt5_13b_mmarco_100k = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoT5Reranker,
        model_name_or_path="unicamp-dl/mt5-13b-mmarco-100k",
        fp_options="float16",
    ),
    name="unicamp-dl/mt5-13b-mmarco-100k",
    languages=mt5_languages,
    open_weights=True,
    revision="e1a4317e102a525ea9e16745ad21394a4f1bffbc",
    release_date="2022-11-04",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["PyTorch"],
    is_cross_encoder=True,
)
