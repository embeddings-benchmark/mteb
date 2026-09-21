"""Run MOEB evaluations.

Fill in MODELS and TASKS below, then run:

    uv sync --extra audio --extra image --extra video
    python scripts/run_moeb.py

Some models need extra dependencies; running one will say which are missing, or
set MTEB_AUTO_INSTALL_EXTRAS=1 to install them as needed.

Results go to the mteb cache (MTEB_CACHE, default ~/.cache/mteb). Tasks that
already have results are skipped, so a run can be stopped and resumed. To submit:

    python -c "import mteb; mteb.ResultCache().submit_results('<model>')"
"""

import mteb

MODELS = [
    "laion/clap-htsat-fused",
    "openai/whisper-tiny",
]

TASKS = [
    "ESC50",
    "GTZANGenre",
    "GunshotTriangulation",
]

BATCH_SIZE = 8

failed = []
for model_name in MODELS:
    model = mteb.get_model(model_name)
    for task in mteb.get_tasks(tasks=TASKS):
        try:
            mteb.evaluate(
                model,
                [task],
                encode_kwargs={"batch_size": BATCH_SIZE},
                raise_error=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad task should not stop the run
            failed.append(f"{model_name} {task.metadata.name}: {exc}")

for line in failed:
    print("FAILED", line)
