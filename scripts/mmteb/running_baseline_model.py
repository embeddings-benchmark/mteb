import mteb


def _name_to_path(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


baseline_models = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]

tasks = mteb.get_tasks()

# potentially create slurm jobs instead
for model_name in baseline_models:
    model = mteb.get_model(model_name)
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=f"results/{_name_to_path(model_name)}")
