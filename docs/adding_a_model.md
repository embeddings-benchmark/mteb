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
tasks = mteb.get_benchmark("MTEB(eng, classic)") # or use a specific benchmark

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model, output_folder="results")
```

Or using the command line interface:

```bash
mteb run -m {model_name} -t {task_names}
```

These will save the results in a folder called `results/{model_name}/{model_revision}`.

1. **Format the results using the CLI:**

```bash
mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md
```

If readme of model exists:

```bash
mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md --from_existing your_existing_readme.md 
```

2. **Add the frontmatter to model repository:**

Copy the content of the `model_card.md` file to the top of a `README.md` file of your model on the Hub. See [here](https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit/blob/main/README.md) for an example.

3. **Wait for a refresh the leaderboard:**

The leaderboard [automatically refreshes daily](https://github.com/embeddings-benchmark/leaderboard/commits/main/) so once submitted you only need to wait for the automatic refresh. You can find the workflows for the leaderboard refresh [here](https://github.com/embeddings-benchmark/leaderboard/tree/main/.github/workflows). If you experience issues with the leaderboard please create an [issue](https://github.com/embeddings-benchmark/mteb/issues).

**Notes:**
- We remove models with scores that cannot be reproduced, so please ensure that your model is accessible and scores can be reproduced.
- An alternative way of submitting to the leaderboard is by opening a PR with your results [here](https://github.com/embeddings-benchmark/results) & checking that they are displayed correctly by [locally running the leaderboard](https://github.com/embeddings-benchmark/leaderboard?tab=readme-ov-file#developer-setup)

- ##### Using Prompts with Sentence Transformers

    If your model uses Sentence Transformers and requires different prompts for encoding the queries and corpus, you can take advantage of the `prompts` [parameter](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer). 
    
    Internally, `mteb` uses the prompt named `query` for encoding the queries and `passage` as the prompt name for encoding the corpus. This is aligned with the default names used by Sentence Transformers.

    ###### Adding the prompts in the model configuration (Preferred)

    You can directly add the prompts when saving and uploading your model to the Hub. For an example, refer to this [configuration file](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5/blob/3b5a16eaf17e47bd997da998988dce5877a57092/config_sentence_transformers.json).

    ###### Instantiating the Model with Prompts

    If you are unable to directly add the prompts in the model configuration, you can instantiate the model using the `sentence_transformers_loader` and pass `prompts` as an argument. For more details, see the `mteb/models/bge_models.py` file.