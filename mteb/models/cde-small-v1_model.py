import mteb

# load a model from the hub (or for a custom implementation see https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md)


model = mteb.get_model(
    "jxm/cde-small-v1",
    trust_remote_code=True,
    model_prompts={
        "query": "search_query: ",
        "passage": "search_document: "
    }
)

tasks = mteb.get_benchmark("MTEB(eng, classic)") # or use a specific benchmark

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model, output_folder="results")


