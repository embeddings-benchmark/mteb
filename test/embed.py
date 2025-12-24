import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


if __name__ == "__main__":
    import mteb
    from mteb.models.vllm_wrapper import vllm_loader

    MODEL_NAME = "intfloat/e5-small"
    MAIN_SCORE = 0.7422994752439667
    MTEB_EMBED_TASKS = ["STS12"]
    MTEB_EMBED_TOL = 1e-4

    encoder = vllm_loader(model_name=MODEL_NAME)
    tasks = mteb.get_tasks(tasks=MTEB_EMBED_TASKS)

    results = mteb.evaluate(
        encoder,
        tasks,
        cache=None,
        show_progress_bar=False,
    )

    vllm_main_score = results[0].scores["test"][0]["main_score"]
    print("ST:", MAIN_SCORE)
    print("vllm:", vllm_main_score)

    assert MAIN_SCORE - vllm_main_score < MTEB_EMBED_TOL

    encoder.cleanup()
