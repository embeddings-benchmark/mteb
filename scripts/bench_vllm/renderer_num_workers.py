"""
Benchmark renderer_num_workers using VllmEncoderWrapper.

For small embedding models (e.g. BAAI/bge-m3), the tokenization and
preprocessing stage often becomes the throughput bottleneck.

- Use multithreading to accelerate preprocessing. You can specify the
number of threads using renderer_num_workers. The total time scales
down almost linearly as the number of renderer workers increases,
if you encounter preprocessing bottlenecks.
- Tiling to overlap preprocessing and computation for pooling models
offline inference. When preprocessing takes less time than computation,
the preprocessing overhead can be almost entirely overlapped.
"""

import os
import time
import argparse
import torch
import gc
from mteb.models.vllm_wrapper import VllmEncoderWrapper


os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def run_benchmark(
    model: str,
    workers_list: list[int],
    prompt_lengths: list[int],
    n_prompt: int = 100,
    max_num_seqs: int = 8,
    dtype: str = "float16",
):
    results = {}
    for workers in workers_list:
        print(f"\n=== renderer_num_workers={workers} ===")
        encoder = VllmEncoderWrapper(
            model=model,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.8,
            renderer_num_workers=workers,
        )
        llm = encoder.llm

        latencies = {}
        for length in prompt_lengths:
            prompt = "hello " * length
            prompts = [prompt] * n_prompt

            # Warmup
            _ = llm.embed(
                prompts[:10],
                tokenization_kwargs={"truncate_prompt_tokens": 512},
                use_tqdm=False,
            )

            # Timed run
            torch.cuda.synchronize()
            start = time.perf_counter()
            llm.embed(
                prompts,
                tokenization_kwargs={"truncate_prompt_tokens": 512},
                use_tqdm=False,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            latencies[length] = elapsed
            print(f"  Prompt length {length:4d}: {elapsed:.4f} sec")

        results[workers] = latencies
        # Clean up
        encoder.cleanup()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_table(results, prompt_lengths, n_prompt):
    workers = sorted(results.keys())
    print("\n### End‑to‑end Latency (seconds) for 100 prompts")
    header = ["Prompt Length"] + [f"workers={w}" for w in workers]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for length in prompt_lengths:
        row = [str(length)]
        base = results[workers[0]][length]
        for w in workers:
            t = results[w][length]
            speedup = base / t if t > 0 else 0
            row.append(f"{t:.4f} ({speedup:.2f}x)")
        print("| " + " | ".join(row) + " |")


def plot_results(results, prompt_lengths, output_file=None):
    if not HAS_PLT:
        print("matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for w, latencies in results.items():
        y = [latencies[l] for l in prompt_lengths]
        ax.plot(prompt_lengths, y, marker="o", label=f"workers={w}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Prompt Length (words, log2 scale)")
    ax.set_ylabel("Time for 100 embeddings (s, log10 scale)")
    ax.set_title("Effect of renderer_num_workers on Preprocessing Speed")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark renderer_num_workers via VllmEncoderWrapper."
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--prompt-lengths", nargs="+", type=int, default=[2**i for i in range(4, 18)]
    )  # 16..4096
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--save-plot", type=str, default="renderer_workers.png")
    args = parser.parse_args()

    results = run_benchmark(
        model=args.model,
        workers_list=args.workers,
        prompt_lengths=args.prompt_lengths,
        n_prompt=args.n_prompts,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
    )

    print_table(results, args.prompt_lengths, args.n_prompts)
    plot_results(results, args.prompt_lengths, args.save_plot)


if __name__ == "__main__":
    main()
