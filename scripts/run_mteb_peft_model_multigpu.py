# To run this script on multiple GPUs, you need to install the following branch of BEIR
# pip install git+https://github.com/NouamaneTazi/beir@nouamane/better-multi-gpu

# Then use this command to run on 2 GPUs for example
# torchrun --nproc_per_node=2 run_mteb_peft_model_multigpu.py

import json
import logging
import os
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from mteb import MTEB
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


BATCH_SIZE = 256
CORPUS_CHUNK_SIZE = 16
HF_MODEL_PATH = "THUDM/chatglm2-6b"
MAX_SEQ_LENGTH = 256
HF_PEFT_MODEL_PATH = "oliverwang15/FinGPT_v31_ChatGLM2_Sentiment_Instruction_LoRA_FT"


class DirectLoadTransformer(nn.Module):
    """
    This is a class modified from the original sentence_transformers's Transformer object.
    The init function is changed to load PEFT models.
    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param peft_model_path: Huggingface PEFT models name (https://huggingface.co/models)
    """

    def __init__(
        self,
        model_name_or_path: str,
        peft_model_path: str,
        max_seq_length: int = 8,
    ):
        super(DirectLoadTransformer, self).__init__()

        # 4 bit quantization ref: https://github.com/AI4Finance-Foundation/FinGPT/blob/master/fingpt/FinGPT-v3/training_int4/train.ipynb
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # please note, because "target_device" is defaulted to None, SentenceTransformer will use multiple GPUs to encode sentences.
        # we don't want to mess up with `device_map="auto"`, that is needed for accelerate inference
        # see multiple GPUs for sentence transformers here:
        # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search_multi_gpu.py#L62-L64
        # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search_multi_gpu.py#L126-L127
        # reference for `device_map="auto"` https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference#loading-weights
        self.auto_model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            quantization_config=q_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            truncation=True,
            max_lenght=max_seq_length,
        )

        self.auto_model = PeftModel.from_pretrained(self.auto_model, peft_model_path)
        self.auto_model = self.auto_model.eval()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = False
        self.max_seq_length = max_seq_length  # this controls the tokens truncation

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(
            self.get_config_dict(), self.auto_model.__class__.__name__
        )

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            }
        )

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length
            )
        )
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                breakpeft_model_path

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return DirectLoadTransformer(model_name_or_path=input_path)


class DirectLoadSentenceTransformer(SentenceTransformer):
    """
    This is a modification of SentenceTransformer object, where is extended to use PEFT models.
    """

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning(
            "No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(
                model_name_or_path
            )
        )
        transformer_model = DirectLoadTransformer(
            model_name_or_path=model_name_or_path,
            peft_model_path=HF_PEFT_MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(), "mean"
        )
        return [transformer_model, pooling_model]


if __name__ == "__main__":
    dist.init_process_group("nccl")
    device_id = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(torch.cuda.device(device_id))

    # Enable logging only first rank=0
    rank = int(os.getenv("RANK", 0))
    if rank != 0:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)
    # breakpoint()
    model = DirectLoadSentenceTransformer(HF_MODEL_PATH)
    eval = MTEB(tasks=["FiQA2018"])

    # ref: https://github.com/embeddings-benchmark/mteb/blob/4d75ddf448c93b4b879e60e110061f7dcf76ae42/mteb/abstasks/AbsTaskRetrieval.py#L15
    eval.run(
        model,
        batch_size=BATCH_SIZE,
        corpus_chunk_size=CORPUS_CHUNK_SIZE,
        score_function="cos_sim",
        overwrite_results=True,
        eval_splits=["test"],
    )
