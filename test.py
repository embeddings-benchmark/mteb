import mteb
print('mteb imported.')

model_name = "facebook/wav2vec2-xls-r-300m"
model = mteb.get_model(model_name=model_name)
print('model loaded..')
tasks = mteb.get_tasks(tasks=["FSD50K"])
print('task loaded..')
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
print('eval complete.')
print(type(results))