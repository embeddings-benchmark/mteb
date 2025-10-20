# Tasks

A task is an implementation of a dataset for evaluation. It could, for instance, be the MIRACL dataset consisting of queries, a corpus of documents
,and the correct documents to retrieve for a given query. In addition to the dataset, a task includes the specifications for how a model should be run on the dataset and how its output should be evaluated. Each task also comes with extensive metadata including the license, who annotated the data, etc.

<figure markdown="span">
    ![](../images/visualizations/task_explainer.png){ width="80%" }
    <figcaption>An overview of the tasks within `mteb`</figcaption>
</figure>

## Utilities

:::mteb.get_tasks

:::mteb.get_task

:::mteb.filter_tasks

## Metadata

Each task also contains extensive metadata. We annotate this using the following object, which allows us to use [pydantic](https://docs.pydantic.dev/latest/) to validate the metadata.

:::mteb.TaskMetadata

## Metadata Types

:::mteb.abstasks.task_metadata.AnnotatorType

:::mteb.abstasks.task_metadata.SampleCreationMethod

:::mteb.abstasks.task_metadata.TaskCategory

:::mteb.abstasks.task_metadata.TaskDomain

:::mteb.abstasks.task_metadata.TaskType

:::mteb.abstasks.task_metadata.TaskSubtype

:::mteb.abstasks.task_metadata.PromptDict


## The Task Object

All tasks in `mteb` inherits from the following abstract class.


:::mteb.AbsTask

## Multimodal Tasks

Tasks that support any modality (text, image, etc.) inherit from the following abstract class. Retrieval tasks support multimodal input (e.g. image + text queries and image corpus or vice versa).

:::mteb.abstasks.retrieval.AbsTaskRetrieval

:::mteb.abstasks.retrieval_dataset_loaders.RetrievalSplitData
    options:
        show_root_toc_entry: false

:::mteb.abstasks.classification.AbsTaskClassification

:::mteb.abstasks.multilabel_classification.AbsTaskMultilabelClassification

:::mteb.abstasks.clustering.AbsTaskClustering

:::mteb.abstasks.sts.AbsTaskSTS

:::mteb.abstasks.zeroshot_classification.AbsTaskZeroShotClassification

:::mteb.abstasks.regression.AbsTaskRegression

:::mteb.abstasks.clustering_legacy.AbsTaskClusteringLegacy

## Text Tasks

:::mteb.abstasks.text.bitext_mining.AbsTaskBitextMining

:::mteb.abstasks.pair_classification.AbsTaskPairClassification

:::mteb.abstasks.text.summarization.AbsTaskSummarization

:::mteb.abstasks.text.reranking.AbsTaskReranking

## Image Tasks

:::mteb.abstasks.image.image_text_pair_classification.AbsTaskImageTextPairClassification
