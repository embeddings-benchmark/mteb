
<!-- If you are not submitting for a dataset, feel free to remove the content below  -->


<!-- add additional description, question etc. related to the new dataset -->

## Checklist for adding MMTEB dataset

<!-- 
Before you commit here is a checklist you should complete before submitting
if you are not 
 -->
Reason for dataset addition:
<!-- Add reason for adding dataset here. E.g. it covers task/language/domain previously not covered -->


- [ ] I have tested that the dataset runs with the `mteb` package.
- [ ] I have run the following models on the task (adding the results to the pr). These can be run using the `mteb -m {model_name} -t {task_name}` command.
  - [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial (both models gain close to perfect scores) nor random (both models gain close to random scores).
- [ ] If the dataset is too big (e.g. >2048 examples), considering using `self.stratified_subsampling() under dataset_transform()`
- [ ] I have filled out the metadata object in the dataset file (find documentation on it [here](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_dataset.md#2-creating-the-metadata-object)).
- [ ] Run tests locally to make sure nothing is broken using `make test`. 
- [ ] Run the formatter to format the code using `make lint`. 
- [ ] I have added points for my submission to the [points folder](https://github.com/embeddings-benchmark/mteb/blob/main/docs/mmteb/points.md) using the PR number as the filename (e.g. `438.jsonl`).
