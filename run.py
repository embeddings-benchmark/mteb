import mteb
model_name = "Qwen/Qwen2-Audio-7B"
model = mteb.get_model(model_name=model_name)
print("model loaded")
tasks = mteb.get_tasks(tasks=["BeijingOpera"])
print("task loaded")
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
print("eval complete")