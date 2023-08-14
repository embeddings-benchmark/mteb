import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB
import torch.distributed as dist
import torch
import os


# To run this script on multiple GPUs, you need to install the following branch of BEIR
# pip install git+https://github.com/NouamaneTazi/beir@nouamane/better-multi-gpu

# Then use this command to run on 2 GPUs for example
# torchrun --nproc_per_node=2 scripts/retrieval_multigpu.py

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

    model = SentenceTransformer("intfloat/e5-large", device="cuda")
    # eval = MTEB(tasks=["MSMARCO"])
    eval = MTEB(task_types=["Retrieval"])
    eval.run(model, batch_size=1024, overwrite_results=True, eval_splits=["test"])
