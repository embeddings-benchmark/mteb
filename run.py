import mteb

task = mteb.get_task("Vidore2ESGReportsHLRetrieval")
model = mteb.get_model("nomic-ai/nomic-embed-multimodal-3b")

res = mteb.evaluate(model, task)
print(res.get_score())