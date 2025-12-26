import time

import mteb
from mteb.models.vllm_wrapper import VllmEncoderWrapper


def get_results(model: str, tasks: list[str]):
    """Evaluate a model on specified MTEB tasks using vLLM for inference."""
    # Auto downcasting torch.float32 to torch.float16.
    encoder = VllmEncoderWrapper(model=model)
    tasks = mteb.get_tasks(tasks=tasks)

    start = time.perf_counter()
    # It only take 2.725 GB of gpu and completes the computation in just 6 minutes.
    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )
    end = time.perf_counter()
    elapsed_time = end - start
    main_score = results[0].scores["dev"][0]["main_score"]
    # elapsed_time 372.5035745330001 main_score 0.66899
    print("elapsed_time", elapsed_time, "main_score", main_score)
    return results


if __name__ == "__main__":
    model = "BAAI/bge-m3"
    tasks = ["T2Reranking"]

    results = get_results(model=model, tasks=tasks)

