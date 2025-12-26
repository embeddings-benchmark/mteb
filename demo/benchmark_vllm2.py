import os

import torch

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.counter import Counter
import gc


def benchmark_vllm(args):
    for batchsize in args.batchsize:
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            max_num_seqs=batchsize,
            max_num_batched_tokens=batchsize * args.max_model_len * 2,
            enforce_eager=args.enforce_eager,
        )
        llm.n_step = 0

        llm_engine_step = llm.llm_engine.step

        def step():
            llm.n_step += 1
            return llm_engine_step()

        llm.llm_engine.step = step

        def warmup(prompts):
            time.sleep(2)
            outputs = llm.embed(prompts[:10], use_tqdm=False)
            assert len(outputs[0].prompt_token_ids) == input_len

        def run(prompts):
            time.sleep(2)

            llm.n_step = 0
            llm.request_counter = Counter()
            start = time.perf_counter()
            outputs = llm.embed(prompts, use_tqdm=False)
            end = time.perf_counter()
            assert len(outputs[-1].prompt_token_ids) == input_len

            n_step = llm.n_step
            elapsed_time = end - start
            delay = elapsed_time / n_step

            print(
                f"Batchsize {batchsize}, Input_len {input_len} Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {delay * 1000:0.2f} ms, n_step {n_step}"
            )

        for input_len in args.input_len:
            prompt = "你" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            warmup(prompts)
            run(prompts)

        del llm, llm_engine_step
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-m3"
    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 1024
    args.num_prompts = 10000
    args.enforce_eager = False
    args.batchsize = [1, 2, 4, 8, 16, 32, 64, 128]
    args.input_len = [512]

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    for dtype in ["float16", "float32"]:
        args.dtype = dtype
        run(args)
