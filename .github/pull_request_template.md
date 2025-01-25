
<!-- If you are submitting a dataset or a model for the model registry please use the corresponding checklists below otherwise feel free to remove them. -->

<!-- add additional description, question etc. related to the new dataset -->


### Code Quality
<!-- Please do not delete this -->
- [ ] **Tests Passed**: Run tests locally using `make test` or `make test-with-coverage` to ensure no existing functionality is broken.
- [ ] **Code Formatted**: Format the code using `make lint` to maintain consistent style.

### Documentation
- [ ] **Updated Documentation**: Add or update documentation to reflect the changes introduced in this PR.

### Testing
- [ ] **New Tests Added**: Write tests to cover new functionality. Validate with `make test-with-coverage`.


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
   - [ ] `mteb.get_model(model_name, revision)` and
   - [ ] `mteb.get_model_meta(model_name, revision)`
 - [ ] I have tested the implementation works on a representative set of tasks.