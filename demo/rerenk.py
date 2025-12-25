import tempfile

import mteb
from mteb.models.vllm_wrapper import VllmCrossEncoderWrapper


def get_results(model_name: str, tasks: list[str], languages: list[str]):
    """Evaluate a model on specified MTEB tasks using vLLM for inference."""
    cross_encoder = VllmCrossEncoderWrapper(model_name=model_name)

    with tempfile.TemporaryDirectory() as prediction_folder:
        bm25s = mteb.get_model("bm25s")
        eval_splits = ["test"]

        mteb_tasks = mteb.get_tasks(
            tasks=tasks, languages=languages, eval_splits=eval_splits
        )

        mteb.evaluate(
            bm25s,
            mteb_tasks,
            prediction_folder=prediction_folder,
            show_progress_bar=False,
            # don't save results for test runs
            cache=None,
            overwrite_strategy="always",
        )

        second_stage_tasks = []
        for task in mteb_tasks:
            second_stage_tasks.append(
                task.convert_to_reranking(
                    prediction_folder,
                    top_k=10,
                )
            )

        results = mteb.evaluate(
            cross_encoder,
            second_stage_tasks,
            show_progress_bar=False,
            cache=None,
        )
    return results


if __name__ == "__main__":
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MTEB_RERANK_TASKS = ["NFCorpus"]
    MTEB_RERANK_LANGS = ["eng"]
    MTEB_RERANK_TOL = 2e-3
    MAIN_SCORE = 0.33457

    results = get_results(
        model_name=MODEL_NAME, tasks=MTEB_RERANK_TASKS, languages=MTEB_RERANK_LANGS
    )

    vllm_main_score = results[0].scores["test"][0]["main_score"]
    print("ST:", MAIN_SCORE)
    print("vllm:", vllm_main_score)

    assert MAIN_SCORE - vllm_main_score < MTEB_RERANK_TOL
