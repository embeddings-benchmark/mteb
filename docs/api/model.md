# Models

A model in `mteb` covers two concepts: metadata and implementation.
- Metadata contains information about the model such as maximum input
length, valid frameworks, license, and degree of openness.
- Implementation is a reproducible workflow, which allows others to run the same model again, using the same prompts, hyperparameters, aggregation strategies, etc.

<figure markdown="span">
    ![](../images/visualizations/modelmeta_explainer.png){ width="80%" }
    <figcaption>An overview of the model and its metadata within `mteb`</figcaption>
</figure>


## Utilities

:::mteb.get_model_metas

:::mteb.get_model_meta

:::mteb.get_model


## Metadata

:::mteb.models.model_meta.ModelMeta

## Model Protocols

:::mteb.models.EncoderProtocol

:::mteb.models.SearchProtocol

:::mteb.models.CrossEncoderProtocol

:::mteb.models.MTEBModels
