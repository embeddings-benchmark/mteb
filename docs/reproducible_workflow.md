# Reproducible Workflows

This section introduce how to MTEB uses reproducible workflows. The main goal is to make the results reproducible and transparent. The workflow is based on the following principles:

1. **Version control**: Both code and data are versioned using a git revision id's referring to a specific version of the model and data.
2. **Model Repository**: MTEB includes a model repository of models run on MTEB since June 2024. These implementation are stored in `mteb/models/`. This is to ensure that the way the model is run is transparent and reproducible. Note that models which is simply loaded and run using SentenceTransformers, are not documented as referring to the revision id of the model is sufficient to reproduce the results.
3. **Result Reproducibility**: Results within MTEB are expected to be reproducible up to the 3rd decimal point.

The general workflow of using a reproducible workflow is as follows:

```{python}
import mteb

model_name = "intfloat/multilingual-e5-small"
revision = "4dc6d853a804b9c8886ede6dda8a073b7dc08a81"

model = mteb.get_model("intfloat/multilingual-e5-small", revision_id=revision)

tasks = mteb.get_tasks(tasks = ["MIRACLReranking"], languages = ["eng"])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)
```

This workflow should produce the produces the same results as the original run. The results are by default stored in `results/{model_name}/{revision_id}/{task_name}.json`.

## Adding a model to the model repository

To add a model to the model repository, the following steps should be followed:

1. **Add a ModelMeta**

Add a ModelMeta object to `mteb/models/*`. This object among other things contain:
    - `model_name`: The name of the model
    - `revision`: The revision id of the model
    - `languages`: The languages the model is trained on
    - ...
  
You might addionally want to specify things like whether the model is open source, framework, etc.

1. **If you model is not a SentenceTransformer compatible**

Additionally specify the `loader` in the ModelMeta object. This is a function that loads the model and returns a mteb compatible `Encoder` model.

1. **Submit a pull request**

Submit a pull request with the new model. The model will be reviewed and added to the model repository.