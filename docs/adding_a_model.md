## Adding a Model to the MTEB Leaderboard

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard). To submit to it:

1. **Run the desired model on MTEB:**

Either use the Python API:

```python
import mteb

# load a model from the hub (or for a custom implementation see https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md)
model = mteb.get_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

tasks = mteb.get_tasks(...) # get specific tasks
# or 
from mteb.benchmarks import MTEB_MAIN_EN
tasks = MTEB_MAIN_EN # or use a specific benchmark

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model, output_folder="results")
```

Or using the command line interface:

```bash
mteb run -m {model_name} -t {task_names}
```

These will save the results in a folder called `results/{model_name}/{model_revision}`.

For reference you can also look at [scripts/run_mteb_english.py](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py) for all MTEB English datasets used in the main ranking, or [scripts/run_mteb_chinese.py](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_chinese.py) for the Chinese ones. 
Advanced scripts with different models are available in the [mteb/mtebscripts repo](https://github.com/embeddings-benchmark/mtebscripts).

2. **Format the results using the CLI:**

```bash
mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md
```

If readme of model exists:

```bash
mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md --from_existing your_existing_readme.md 
```

3. **Add the frontmatter to model repository:**

Copy the content of the `model_card.md` file to the top of a `README.md` file of your model on the Hub. See [here](https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit/blob/main/README.md) for an example.

4. **Wait for a refresh the leaderboard:**

The leaderboard will then automatically refresh daily so once submitted all you have to do is wait for the automatic refresh.

You can find the workflows for the leaderboard refresh [here](https://github.com/embeddings-benchmark/leaderboard/tree/main/.github/workflows). If you experience issues with the leaderboard please create an [issue](https://github.com/embeddings-benchmark/mteb/issues).
