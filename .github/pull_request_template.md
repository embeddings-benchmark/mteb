
<!-- If you are not submitting for a dataset, feel free to remove the content below  -->


<!-- add additonal description, question etc. related to the new dataset -->

## Checklist for adding MMTEB dataset
<!-- 
Before you commit here is a checklist you should complete before submitting
if you are not 
 -->

- [ ] I have tested that the dataset runs with the `mteb` package.
- [ ] I have run the following models on the task (adding the results to the pr). These can be run using the `mteb run -m {model_name} -t {task_name}` command.
  - [ ] `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `intfloat/multilingual-e5-small`
- [ ] I have checked that the performance is neither trivial (both models gain close to perfect scores) nor random (both models gain close to random scores).
- [ ] I have considered the size of the dataset and reduced it if it is too big (2048 examples is typically large enough for most tasks)
- [ ] Run tests locally to make sure nothing is broken using `make test`. 
- [ ] Run the formatter to format the code using `make lint`. 
- [ ] I have added points for my submission to the [POINTS.md](https://github.com/embeddings-benchmark/mteb/blob/main/docs/mmteb/points.md) file.