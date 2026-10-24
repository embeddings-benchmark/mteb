"""
vLLM vs HuggingFace Embedding Benchmark

This script benchmarks the inference performance of two embedding backends:
- HuggingFace (via sentence-transformers)
- vLLM (via mteb's VllmEncoderWrapper)

It measures throughput (requests/sec and tokens/sec) and average batch latency
under varying data types (float16, bfloat16, float32), batch sizes, and input
lengths.
"""

import os
import time
import gc
import torch
import argparse
import vllm
from concurrent.futures import ProcessPoolExecutor

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import sentence_transformers
except ImportError:
    print("Warning: sentence-transformers not installed.")
try:
    from vllm import LLM
except ImportError:
    print("Warning: vLLM not installed.")


def get_system_info():
    info = []
    info.append(f"PyTorch {torch.__version__}")
    info.append(f"vLLM {vllm.__version__}")
    if torch.cuda.is_available():
        info.append(f"CUDA {torch.version.cuda}")
        info.append(f"GPU {torch.cuda.get_device_name(0)}")
    return ", ".join(info)


def check_dtype_support(dtype):
    """Check if the given dtype is supported on the current GPU."""
    if dtype == "bfloat16":
        if not torch.cuda.is_available():
            return False
        # bfloat16 requires compute capability >= 8.0
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            print(f"Warning: bfloat16 not supported on this GPU.")
            return False
    return True


def benchmark_hf(args):
    """Benchmark HuggingFace/SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    results = {}

    for dtype in args.dtypes:
        if not check_dtype_support(dtype):
            print(f"Skipping dtype {dtype} for HF due to lack of hardware support.")
            continue

        print(f"\n=== Benchmarking HF with dtype: {dtype} ===")
        model = SentenceTransformer(
            args.model,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
            trust_remote_code=True,
        )

        dtype_results = []

        with torch.no_grad():
            for batchsize in args.batchsize:
                batch_results = {}
                for input_len in args.input_len:
                    prompt = "hello " * (input_len // 2 - 1)
                    requests = [prompt for _ in range(args.num_prompts)]

                    inputs_batch = model.tokenizer(prompt)
                    assert len(inputs_batch["input_ids"]) == input_len

                    # Warmup
                    model.encode(requests[:10], batch_size=batchsize)
                    torch.cuda.synchronize()

                    start = time.perf_counter()

                    n_step = 0
                    for i in range(0, len(requests), batchsize):
                        batch = requests[i : i + batchsize]
                        model.encode(batch, batch_size=batchsize)
                        n_step += 1

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    elapsed_time = end - start
                    delay = elapsed_time / n_step * 1000
                    throughput_req = len(requests) / elapsed_time
                    throughput_tokens = (len(requests) * input_len) / elapsed_time

                    batch_results[input_len] = {
                        "throughput_req": throughput_req,
                        "throughput_tokens": throughput_tokens,
                        "latency": delay,
                        "n_step": n_step,
                        "elapsed_time": elapsed_time,
                    }

                    print(
                        f"  Batchsize {batchsize}, Input_len {input_len}: "
                        f"Throughput: {throughput_req:.4f} req/s, "
                        f"{throughput_tokens:.4f} tokens/s, "
                        f"Latency (batch): {delay:.2f} ms"
                    )

                dtype_results.append(batch_results)

            results[dtype] = dtype_results

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def benchmark_vllm(args):
    """Benchmark vLLM model using VllmEncoderWrapper (mteb)"""
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.utils.counter import Counter
    from mteb.models.vllm_wrapper import VllmEncoderWrapper

    results = {}

    for dtype in args.dtypes:
        if not check_dtype_support(dtype):
            print(f"Skipping dtype {dtype} for vLLM due to lack of hardware support.")
            continue

        print(f"\n=== Benchmarking vLLM with dtype: {dtype} ===")
        dtype_results = []

        for batchsize in args.batchsize:
            batch_results = {}
            encoder = None
            llm = None
            original_step = None

            try:
                encoder = VllmEncoderWrapper(
                    model=args.model,
                    dtype=dtype,
                    max_model_len=args.max_model_len * 2,
                    max_num_seqs=batchsize,
                    max_num_batched_tokens=batchsize * args.max_model_len * 2,
                    gpu_memory_utilization=0.8,
                )

                llm = encoder.llm
                llm.n_step = 0
                original_step = llm.llm_engine.step

                def step():
                    llm.n_step += 1
                    return original_step()

                llm.llm_engine.step = step

                for input_len in args.input_len:
                    prompt = "hello " * (input_len // 2 - 1)
                    prompts = [prompt for _ in range(args.num_prompts)]

                    # Warmup
                    time.sleep(2)
                    outputs = llm.embed(prompts[:10], use_tqdm=False)
                    assert len(outputs[0].prompt_token_ids) == input_len

                    # Run benchmark
                    time.sleep(2)
                    llm.n_step = 0
                    llm.request_counter = Counter()
                    start = time.perf_counter()
                    outputs = llm.embed(prompts, use_tqdm=False)
                    end = time.perf_counter()
                    assert len(outputs[-1].prompt_token_ids) == input_len

                    n_step = llm.n_step
                    elapsed_time = end - start
                    delay = elapsed_time / n_step * 1000
                    throughput_req = len(prompts) / elapsed_time
                    throughput_tokens = (len(prompts) * input_len) / elapsed_time

                    batch_results[input_len] = {
                        "throughput_req": throughput_req,
                        "throughput_tokens": throughput_tokens,
                        "latency": delay,
                        "n_step": n_step,
                        "elapsed_time": elapsed_time,
                    }

                    print(
                        f"  Batchsize {batchsize}, Input_len {input_len}: "
                        f"Throughput: {throughput_req:.4f} req/s, "
                        f"{throughput_tokens:.4f} tokens/s, "
                        f"Latency (batch): {delay:.2f} ms"
                    )

                dtype_results.append(batch_results)

            except Exception as e:
                print(f"  Error with batchsize {batchsize}: {e}")
                for input_len in args.input_len:
                    batch_results[input_len] = None
                dtype_results.append(batch_results)

            finally:
                del original_step, llm
                encoder.cleanup()
                del encoder

        results[dtype] = dtype_results

    return results


def print_perf_table(batchsizes, perf_data, input_len):
    """Print a Markdown table showing throughput and latency for each configuration."""
    sys_info = get_system_info()
    print(f"\n**System:** {sys_info}\n")
    print(
        f"### Throughput (tokens/s) and Latency (ms/batch) — Input Length = {input_len}"
    )

    config_names = list(perf_data.keys())
    # Header: Batch Size, then each config: Throughput, Latency
    header = ["Batch Size"]
    for name in config_names:
        header.append(f"{name} Tok/s")
        header.append(f"{name} Latency(ms)")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for i, bs in enumerate(batchsizes):
        row = [str(bs)]
        for name in config_names:
            entry = perf_data[name][i]
            if entry is not None:
                row.append(f"{entry['throughput_tokens']:.2f}")
                row.append(f"{entry['latency']:.2f}")
            else:
                row.append("N/A")
                row.append("N/A")
        print("| " + " | ".join(row) + " |")


def plot_latency_vs_throughput(batchsizes, perf_data, input_len, output_file=None):
    """Plot Throughput (tokens/s) vs Latency (ms/batch) with scatter + lines, Y log scale."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available.")
        return

    sys_info = get_system_info()
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, entries) in enumerate(perf_data.items()):
        x_vals, y_vals = [], []
        for entry in entries:
            if entry is not None:
                x_vals.append(entry["throughput_tokens"])
                y_vals.append(entry["latency"])
        if not x_vals:
            continue

        ax.plot(
            x_vals,
            y_vals,
            color=colors[idx % len(colors)],
            marker="o",
            markersize=6,
            linestyle="-",
            linewidth=2,
            label=name,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("Latency (ms per batch) [log scale]")
    ax.set_title(f"Throughput vs Latency\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison between vLLM and HuggingFace/Transformers."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Model name or path. Default: BAAI/bge-m3",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to benchmark. Default: 1000",
    )
    parser.add_argument(
        "--batchsize",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to test. Default: 1 2 4 8 16 32 64 128",
    )
    parser.add_argument(
        "--input-len",
        nargs="+",
        type=int,
        default=[512],
        help="Input lengths to test. Default: 512",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model length. Default: 1024",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
        help="Data types to test. Default: float16 bfloat16 float32",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=["hf", "vllm", "both"],
        default=["both"],
        help="Which benchmarks to run. Default: both",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not display or save the plot."
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_dtype.png",
        help="Save the plot to the given file path.",
    )
    # --metric argument removed; table and plot now always use throughput_tokens and latency
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Benchmarks will run on CPU (may be slow).")

    backend_tasks = []
    if "hf" in args.benchmark or "both" in args.benchmark:
        try:
            import sentence_transformers

            backend_tasks.append(("HF", benchmark_hf))
        except ImportError:
            print(
                "Error: sentence-transformers is required for HF benchmark. Install with: pip install sentence-transformers"
            )
            return

    if "vllm" in args.benchmark or "both" in args.benchmark:
        try:
            from mteb.models.vllm_wrapper import VllmEncoderWrapper

            backend_tasks.append(("vLLM", benchmark_vllm))
        except ImportError:
            print(
                "Error: mteb is required for vLLM benchmark. Install with: pip install mteb"
            )
            return

    if not backend_tasks:
        print("No valid backends selected. Exiting.")
        return

    print("=" * 60)

    hf_results = None
    vllm_results = None

    with ProcessPoolExecutor(max_workers=1) as executor:
        for idx, (name, func) in enumerate(backend_tasks):
            future = executor.submit(func, args)
            try:
                result = future.result()
                if name == "HF":
                    hf_results = result
                elif name == "vLLM":
                    vllm_results = result
            except Exception as e:
                print(f"Benchmark {name} failed with error: {e}")
                if name == "HF":
                    hf_results = None
                elif name == "vLLM":
                    vllm_results = None

            if idx < len(backend_tasks) - 1:
                time.sleep(1)

    # Build unified performance data structure for all configs
    perf_data = {}
    target_input_len = args.input_len[
        0
    ]  # only first input length used for tables/plots

    if hf_results:
        for dtype, dtype_results in hf_results.items():
            key = f"HF_{dtype}"
            perf_data[key] = []
            for i, bs in enumerate(args.batchsize):
                if (
                    i < len(dtype_results)
                    and dtype_results[i]
                    and target_input_len in dtype_results[i]
                ):
                    metrics = dtype_results[i][target_input_len]
                    perf_data[key].append(
                        {
                            "throughput_tokens": metrics["throughput_tokens"],
                            "latency": metrics["latency"],
                        }
                    )
                else:
                    perf_data[key].append(None)

    if vllm_results:
        for dtype, dtype_results in vllm_results.items():
            key = f"vLLM_{dtype}"
            perf_data[key] = []
            for i, bs in enumerate(args.batchsize):
                if (
                    i < len(dtype_results)
                    and dtype_results[i]
                    and target_input_len in dtype_results[i]
                ):
                    metrics = dtype_results[i][target_input_len]
                    perf_data[key].append(
                        {
                            "throughput_tokens": metrics["throughput_tokens"],
                            "latency": metrics["latency"],
                        }
                    )
                else:
                    perf_data[key].append(None)

    if perf_data:
        print_perf_table(args.batchsize, perf_data, target_input_len)

        if not args.no_plot:
            plot_latency_vs_throughput(
                args.batchsize, perf_data, target_input_len, output_file=args.save_plot
            )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
