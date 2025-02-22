import mteb

from tests.test_benchmark.task_grid import MOCK_MAEB_TASK_GRID
# load a model from the hub (or for a custom implementation see https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md)
model_name = "facebook/wav2vec2-base"
model = mteb.get_model(model_name, revision="main")

# print(model.name)

print("Loaded successfully")

# tasks = mteb.get_tasks(tasks=[MOCK_MAEB_TASK_GRID])
evaluation = mteb.MTEB(tasks=[MOCK_MAEB_TASK_GRID])
results = evaluation.run(model, output_folder=f"results/{model_name}")
