import time


def benchmark_hf(args):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.model,
        model_kwargs={"torch_dtype": args.dtype},
        trust_remote_code=True,
    )

    with torch.no_grad():
        for batchsize in args.batchsize:
            for input_len in args.input_len:
                prompt = "你" * (input_len - 2)
                requests = [prompt for _ in range(args.num_prompts)]

                inputs_batch = model.tokenizer(prompt)
                assert len(inputs_batch["input_ids"]) == input_len

                start = time.perf_counter()

                n_step = 0
                for i in range(0, len(requests), batchsize):
                    batch = requests[i : i + batchsize]
                    model.encode(batch, batch_size=batchsize)
                    n_step += 1

                torch.cuda.synchronize()
                end = time.perf_counter()

                elapsed_time = end - start
                delay = elapsed_time / n_step

                print(
                    f"Batchsize {batchsize}, Throughput: "
                    f"{len(requests) / elapsed_time:.4f} requests/s, "
                    f"{len(requests * input_len) / elapsed_time:.4f} tokens/s, "
                    f"Latency {delay * 1000:0.2f} ms, n_step {n_step}"
                )


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-m3"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 1024
    args.num_prompts = 10000
    args.batchsize = [1, 2, 4, 8, 16, 32, 64, 128]
    args.input_len = [512]

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_hf, args)
            f.result()


    for dtype in ["float16", "float32"]:
        args.dtype = dtype
        run(args)
