# Models

<!-- TODO: Encoder or model? Encoder is consistent with the code, but might be less used WDYT? We also use ModelMeta ... -->

A model in `mteb` covers two concepts: metadata and implementation. 
- Metadata contains information about the model such as maximum input
length, valid frameworks, license, and degree of openness. 
- Implementation is a reproducible workflow, which allows others to run the same model again, using the same prompts, hyperparameters, aggregation strategies, etc.

<figure markdown="span">
    ![](../images/visualizations/modelmeta_explainer.png){ width="80%" }
    <figcaption>An overview of the model and its metadata within `mteb`</figcaption>
</figure>



## Metadata

:::mteb.models.ModelMeta

## The Encoder Interface

:::mteb.Encoder



