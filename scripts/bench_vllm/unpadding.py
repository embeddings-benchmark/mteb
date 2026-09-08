"""
vLLM vs HuggingFace Embedding Benchmark (Variable‑Length Support)

By default, Sentence Transformers (ST) pad all inputs in a batch to the
longest one, which is highly inefficient.  vLLM avoids padding entirely.
This script highlights that ST suffers a noticeable drop in speed with
varied input lengths, while vLLM does not.
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
    """Get system information for reporting."""
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


def generate_prompts(input_len, num_prompts, variable=False):
    """
    Generate a list of prompt strings.

    - variable=False: all requests have the same length, built by repeating "hello ".
    - variable=True : half of the requests are shorter (input_len - half),
                      the other half longer (input_len + half), with half = input_len // 2.
    Returns (requests, total_tokens) where total_tokens is the sum of token counts.
    """
    if variable:
        half = input_len // 2
        short_len = input_len - half
        long_len = input_len + half
        requests = []
        # Alternate short and long requests to create mixed-length batches
        for _ in range(num_prompts // 2):
            requests.append("hello " * (short_len // 2 - 1))
            requests.append("hello " * (long_len // 2 - 1))
        # If num_prompts is odd, add one more long request
        if num_prompts % 2:
            requests.append("hello " * (long_len // 2 - 1))
        total_tokens = (short_len + long_len) * (num_prompts // 2) + (
            long_len if num_prompts % 2 else 0
        )
        return requests, total_tokens
    else:
        # Fixed length
        prompt = "hello " * (input_len // 2 - 1)
        requests = [prompt] * num_prompts
        total_tokens = input_len * num_prompts
        return requests, total_tokens


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
                    modes = [False]  # fixed length always run
                    if args.variable_length:
                        modes.append(True)  # variable length if enabled (default True)

                    for variable in modes:
                        suffix = "_variable-length" if variable else "_fixed-length"
                        key = f"{input_len}{suffix}"
                        requests, total_tokens = generate_prompts(
                            input_len, args.num_prompts, variable
                        )

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
                        throughput_tokens = total_tokens / elapsed_time

                        batch_results[key] = {
                            "throughput_req": throughput_req,
                            "throughput_tokens": throughput_tokens,
                            "latency": delay,
                            "n_step": n_step,
                            "elapsed_time": elapsed_time,
                        }

                        mode_desc = "variable-length" if variable else "fixed-length"
                        print(
                            f"  Batchsize {batchsize}, Input_len {input_len} ({mode_desc}): "
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
                    modes = [False]
                    if args.variable_length:
                        modes.append(True)

                    for variable in modes:
                        suffix = "_variable-length" if variable else "_fixed-length"
                        key = f"{input_len}{suffix}"
                        requests, total_tokens = generate_prompts(
                            input_len, args.num_prompts, variable
                        )

                        # Warmup
                        time.sleep(2)
                        outputs = llm.embed(requests[:10], use_tqdm=False)

                        # Run benchmark
                        time.sleep(2)
                        llm.n_step = 0
                        llm.request_counter = Counter()
                        start = time.perf_counter()
                        outputs = llm.embed(requests, use_tqdm=False)
                        end = time.perf_counter()

                        n_step = llm.n_step
                        elapsed_time = end - start
                        delay = elapsed_time / n_step * 1000
                        throughput_req = len(requests) / elapsed_time
                        throughput_tokens = total_tokens / elapsed_time

                        batch_results[key] = {
                            "throughput_req": throughput_req,
                            "throughput_tokens": throughput_tokens,
                            "latency": delay,
                            "n_step": n_step,
                            "elapsed_time": elapsed_time,
                        }

                        mode_desc = "variable-length" if variable else "fixed-length"
                        print(
                            f"  Batchsize {batchsize}, Input_len {input_len} ({mode_desc}): "
                            f"Throughput: {throughput_req:.4f} req/s, "
                            f"{throughput_tokens:.4f} tokens/s, "
                            f"Latency (batch): {delay:.2f} ms"
                        )

                dtype_results.append(batch_results)

            except Exception as e:
                print(f"  Error with batchsize {batchsize}: {e}")
                for input_len in args.input_len:
                    for suffix in (
                        ["_fixed-length", "_variable-length"]
                        if args.variable_length
                        else ["_fixed-length"]
                    ):
                        batch_results[f"{input_len}{suffix}"] = None
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
    print(f"### Throughput (tokens/s) and Latency (ms/batch)")

    config_names = list(perf_data.keys())
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
        default=["float16"],
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
        "--no-variable-length",
        action="store_true",
        help="Disable variable-length test (default: variable-length is enabled).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not display or save the plot."
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_unpadding.png",
        help="Save the plot to the given file path.",
    )
    args = parser.parse_args()

    # Variable-length is enabled by default, disable with --no-variable-length
    args.variable_length = not args.no_variable_length

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

    perf_data = {}
    target_input_len = args.input_len[0]
    fixed_suffix = "_fixed-length"
    var_suffix = "_variable-length"

    def process_results(backend_results, prefix):
        if backend_results is None:
            return
        for dtype, dtype_results in backend_results.items():
            # Build keys with the new suffixes
            key_fixed = f"{prefix}_{dtype}{fixed_suffix}"
            if key_fixed not in perf_data:
                perf_data[key_fixed] = []
            if args.variable_length:
                key_var = f"{prefix}_{dtype}{var_suffix}"
                if key_var not in perf_data:
                    perf_data[key_var] = []

            for i, bs in enumerate(args.batchsize):
                if i < len(dtype_results) and dtype_results[i]:
                    fixed_key = str(target_input_len) + fixed_suffix
                    if fixed_key in dtype_results[i]:
                        metrics = dtype_results[i][fixed_key]
                        perf_data[key_fixed].append(
                            {
                                "throughput_tokens": metrics["throughput_tokens"],
                                "latency": metrics["latency"],
                            }
                        )
                    else:
                        perf_data[key_fixed].append(None)

                    if args.variable_length:
                        var_key = str(target_input_len) + var_suffix
                        if var_key in dtype_results[i]:
                            metrics = dtype_results[i][var_key]
                            perf_data[key_var].append(
                                {
                                    "throughput_tokens": metrics["throughput_tokens"],
                                    "latency": metrics["latency"],
                                }
                            )
                        else:
                            perf_data[key_var].append(None)
                else:
                    perf_data[key_fixed].append(None)
                    if args.variable_length:
                        perf_data[key_var].append(None)

    process_results(hf_results, "HF")
    process_results(vllm_results, "vLLM")

    if perf_data:
        print_perf_table(args.batchsize, perf_data, target_input_len)

        if not args.no_plot:
            plot_latency_vs_throughput(
                args.batchsize, perf_data, target_input_len, output_file=args.save_plot
            )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
