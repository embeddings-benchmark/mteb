import mteb
from mteb.models.vllm_wrapper import VllmEncoderWrapper


def get_results(model: str, tasks: list[str]):
    encoder = VllmEncoderWrapper(model=model)
    tasks = mteb.get_tasks(tasks=tasks)

    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )
    return results


if __name__ == "__main__":
    MODEL_NAME = "intfloat/e5-small"
    MTEB_EMBED_TASKS = ["STS12"]
    MAIN_SCORE = 0.7422994752439667
    MTEB_EMBED_TOL = 1e-4

    results = get_results(model=MODEL_NAME, tasks=MTEB_EMBED_TASKS)

    vllm_main_score = results[0].scores["test"][0]["main_score"]
    print("ST:", MAIN_SCORE)
    print("vllm:", vllm_main_score)

    assert MAIN_SCORE - vllm_main_score < MTEB_EMBED_TOL
