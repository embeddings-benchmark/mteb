# Reproducible Workflows

This section introduce how to MTEB uses reproducible workflows. The main goal is to make the results reproducible and transparent. The workflow is based on the following principles:

1. **Version control**: Both code and data are versioned using git revision IDs.
2. **Model Repository**: MTEB includes a model registry of models run on MTEB since June 2024. These implementations are stored in `mteb/models/`. This is to ensure that the model run is transparent, documented, and reproducible. Note that models which are simply loaded and run using SentenceTransformers are not documented, as referring only to the revision ID of the model is sufficient to reproduce the results.
3. **Result Reproducibility**: Results within MTEB are expected to be reproducible up to the 3rd decimal point.

Using a reproducible workflow:

```{python}
import mteb

model_name = "intfloat/multilingual-e5-small"
revision = "4dc6d853a804b9c8886ede6dda8a073b7dc08a81"

model = mteb.get_model(model_name, revision_id=revision) # load model using registry implementation if available, otherwise use SentenceTransformers

tasks = mteb.get_tasks(tasks = ["MIRACLReranking"], languages = ["eng"])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```

This workflow should produce the same results as the original run. The results are by default stored in `results/{model_name}/{revision_id}/{task_name}.json`. This approach is equivalent to using the CLI.

## Adding a model to the model registry

To add a model to the model registry, the following steps should be followed:

1. **Add a ModelMeta**

Add a ModelMeta object to `mteb/models/*`. This object among other things contain:
    - `model_name`: The name of the model, e.g. "sentence-transformers/all-MiniLM-L6-v2".
    - `revision`: The revision id of the model
    - `languages`: The list of languages the model is trained on.
    - ...
  
You might addionally want to specify additional parameters like whether the model is open source, framework, etc.

2. **If you model is not a SentenceTransformer compatible**

Additionally specify the `loader` in the ModelMeta object. This is a function that loads the model and returns a mteb compatible `Encoder` model. For the `Encoder` class, see `mteb/encoder_interface.py`.

3. **Submit a pull request**

Submit a pull request with the new model. The model will be reviewed and added to the model repository. Please include the checklist in the pull request:

- [ ] I have filled out the ModelMeta object to the extent possible
- [ ] I have ensure that my model can be loaded using `mteb.get_model(model_name, revision_id)` and `mteb.get_model_meta(model_name, revision_id)`
- [ ] I have tested the implementation works of a representative set of tasks.