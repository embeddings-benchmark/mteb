# Tasks

A task is an implementation of a dataset for evaluation. It could for instance be the MIRACL dataset consisting of queries, a corpus of documents 
as well as the correct documents to retrieve for a given query. In addition to the dataset a task includes specification for how a model should be run on the dataset and how its output should be evaluation. Each task also come with extensive metadata including the license, who annotated the data and so on.

<figure markdown="span">
    ![](../images/visualizations/task_explainer.png){ width="80%" }
    <figcaption>An overview of the tasks within `mteb`</figcaption>
</figure>

## Utilities

:::mteb.get_tasks

:::mteb.get_task

## Metadata

Each task also contains extensive metadata, we annotate this using the following object. This allows to use [pydantic](https://docs.pydantic.dev/latest/) to evaluate that the 
metadata is valid and consistent. 

:::mteb.abstasks.TaskMetadata



## The Task Object

All tasks in `mteb` inherits from the following abstract class.

<!-- 
TODO: we probably need to hide some of the method and potentially add a docstring to the class.
-->

