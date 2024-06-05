
<!-- If you are submitting a dataset or a model for the model registry please use the corresponding checklists below otherwise feel free to remove them. -->

<!-- add additional description, question etc. related to the new dataset -->


## Checklist
<!-- Please do not delete this -->

- [ ] Run tests locally to make sure nothing is broken using `make test`. 
- [ ] Run the formatter to format the code using `make lint`. 


### Adding datasets checklist
<!-- see also https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_dataset.md -->

**Reason for dataset addition**: ... <!-- Add reason for adding dataset here. E.g. it covers task/language/domain previously not covered -->

- [ ] I have run the following models on the task (adding the results to the pr). These can be run using the `mteb -m {model_name} -t {task_name}` command.
  - [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial (both models gain close to perfect scores) nor random (both models gain close to random scores).
- [ ] If the dataset is too big (e.g. >2048 examples), considering using `self.stratified_subsampling() under dataset_transform()`
- [ ] I have filled out the metadata object in the dataset file (find documentation on it [here](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_dataset.md#2-creating-the-metadata-object)).
- [ ] Run tests locally to make sure nothing is broken using `make test`. 
- [ ] Run the formatter to format the code using `make lint`. 


### Adding a model checklist
<!-- 
When adding a model to the model registry
see also https://github.com/embeddings-benchmark/mteb/blob/main/docs/reproducible_workflow.md
-->

 - [ ] I have filled out the ModelMeta object to the extent possible
 - [ ] I have ensured that my model can be loaded using
   - [ ] `mteb.get_model(model_name, revision_id)` and
   - [ ] `mteb.get_model_meta(model_name, revision_id)`
 - [ ] I have tested the implementation works on a representative set of tasks.