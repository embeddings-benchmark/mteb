"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import os
import tqdm
import torch

from mteb.evaluation.MTEB import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks import  ArguAna, SCIDOCS
import huggingface_hub

def log_into_huggingface_hub() -> None:
    """
    Log into the Hugging Face Hub.

    :raises ValueError: If the HUGGINGFACE_TOKEN environment variable is not set.
    """
    
    TOKEN = "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"
    if not TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file.")

    huggingface_hub.login(token=TOKEN, write_permission=True)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

#models = ["intfloat/e5-large-v2","intfloat/multilingual-e5-large","intfloat/multilingual-e5-base","BAAI/bge-large-en-v1.5","sentence-transformers/paraphrase-multilingual-mpnet-base-v2","sentence-transformers/sentence-t5-xl","hackathon-pln-es/paraphrase-spanish-distilroberta","hiiamsid/sentence_similarity_spanish_es","clibrain/retromae_es_67000","clibrain/RetroMAE-finetuned-stsb_multi_es_aug_gpt3.5-turbo_v2","clibrain/multilingual-e5-LARGE-tuned-double-dataset-b16-e5"]
log_into_huggingface_hub()
models = ["clibrain/multilingual-e5-LARGE-stsb-tuned-b16-e5"]#, "bertin-project/bertin-roberta-base-spanish", "dccuchile/bert-base-spanish-wwm-uncased"]
for model_name in tqdm.tqdm(models):
    model = SentenceTransformer(model_name, device="cuda")
    evaluation = MTEB(task_langs=["es"])
    # evaluation = MTEB(tasks=[
    #         SCIDOCS(langs=["en"])
    # ])
    evaluation.run(model, output_folder=f"./results/{model_name}", eval_splits=["test"])
    torch.cuda.empty_cache()

    