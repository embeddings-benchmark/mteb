# pip install git+https://github.com/taeminlee/mteb.git@ontheit 후 사용
# streamlit run leaderboard.py 로 결과 확인

"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from multiprocessing import Process, current_process

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# from dotenv import load_dotenv
from setproctitle import setproctitle

import mteb
from mteb import MTEB, get_tasks
from mteb.encoder_interface import PromptType
from mteb.models.instruct_wrapper import instruct_wrapper
from mteb.models.openai_models import OpenAIWrapper
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.requires_package import requires_package

# import tiktoken

logger = logging.getLogger("main")


class CustomOpenAIWrapper(OpenAIWrapper):
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        requires_package(self, "openai", "Openai text embedding")
        from openai import NotGiven

        if self._model_name == "text-embedding-ada-002" and self._embed_dim is not None:
            logger.warning(
                "Reducing embedding size available only for text-embedding-3-* models"
            )

        trimmed_sentences = []
        for sentence in sentences:
            encoding = tiktoken.get_encoding("cl100k_base")
            encoded_sentence = encoding.encode(sentence)
            if len(encoded_sentence) > 8191:
                trimmed_sentence = encoding.decode(encoded_sentence[:8191])
                trimmed_sentences.append(trimmed_sentence)
            else:
                trimmed_sentences.append(sentence)

        max_batch_size = 2048
        sublists = [
            trimmed_sentences[i : i + max_batch_size]
            for i in range(0, len(trimmed_sentences), max_batch_size)
        ]

        all_embeddings = []

        for sublist in sublists:
            response = self._client.embeddings.create(
                input=sublist,
                model=self._model_name,
                encoding_format="float",
                dimensions=self._embed_dim or NotGiven(),
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)


# load_dotenv()

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument("--quantize", default=False, type=bool, help="quantize embeddings")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = [
    # "Ko-StrategyQA",
    "AutoRAGRetrieval",
    # "MIRACLRetrieval",
    # "PublicHealthQA",
    # "BelebeleRetrieval",
    # "MrTidyRetrieval",
    # "MultiLongDocRetrieval",
    # "XPQARetrieval"
]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


model_names = [
    "BAAI/bge-m3/sparse",  # 8192
]


def evaluate_model(model_name, gpu_id):
    try:
        # Set the environment variable for the specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        model = None
        if not os.path.exists(model_name):
            if "m2v" in model_name:
                static_embedding = StaticEmbedding.from_model2vec(model_name)
                model = SentenceTransformer(modules=[static_embedding])
            else:
                if model_name == "nlpai-lab/KoE5":
                    # mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                        PromptType.passage.value: "passage: ",
                    }
                    model = SentenceTransformerWrapper(
                        model=model_name, model_prompts=model_prompts
                    )
                elif model_name == "BAAI/bge-multilingual-gemma2":
                    instruction_template = "<instruct>{instruction}\n<query>"
                    model = instruct_wrapper(
                        model_name_or_path=model_name,
                        instruction_template=instruction_template,
                        attn="cccc",
                        pooling_method="lasttoken",
                        mode="embedding",
                        torch_dtype=torch.float16,
                        normalized=True,
                    )
                elif "text-embedding-3" in model_name:
                    model = CustomOpenAIWrapper(model_name)
                else:
                    model = mteb.get_model(model_name)
                    # from mteb.models.bge_models import BGEM3Wrapper
                    # model = BGEM3Wrapper(model_name)
        else:
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                if "m2v" in model_name:
                    static_embedding = StaticEmbedding.from_model2vec(model_name)
                    model = SentenceTransformer(modules=[static_embedding])
                else:
                    model = mteb.get_model(model_name)

        if model:
            setproctitle(f"{model_name}-{gpu_id}")
            print(
                f"Running task: {TASK_LIST} / {model_name} on GPU {gpu_id} in process {current_process().name}"
            )
            evaluation = MTEB(
                tasks=get_tasks(
                    tasks=TASK_LIST, languages=["kor-Kore", "kor-Hang", "kor_Hang"]
                )
            )
            # 48GB VRAM 기준 적합한 batch sizes
            if "multilingual-e5" in model_name:
                batch_size = 256
            elif "jina" in model_name:
                batch_size = 8
            elif "bge-m3" in model_name:
                batch_size = 32
            elif "gemma2" in model_name:
                batch_size = 256
            elif "Salesforce" in model_name:
                batch_size = 128
            else:
                batch_size = 64

            evaluation.run(
                model,
                output_folder=f"results/{model_name}",
                encode_kwargs={"batch_size": batch_size},
            )
    except Exception as ex:
        print(ex)
        traceback.print_exc()


if __name__ == "__main__":
    processes = []
    for i, model_name in enumerate(model_names):
        gpu_id = i + 3  # Cycle through available GPUs
        p = Process(target=evaluate_model, args=(model_name, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
