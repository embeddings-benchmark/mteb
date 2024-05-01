## Prompt retrieval evaluation

We use embedding models to retrieve demonstration examples, which are used for LLM in-context learning. The downstream results serve as the metric of embedding models.

### Inference
MultiWOZ
```
python inference_mwoz.py --output_dir {output_dir} --retrieval {retrieval_model}
```
other tasks
```
python inference.py --task {task} --output_dir {output_dir} --batch_size {batch_size} --retrieval {retrieval_model}
```
* --retrieval can be one of: `GritLM/GritLM-7B`, `hkunlp/instructor-base`, `hkunlp/instructor-large`, `hkunlp/instructor-xl`, `sentence-transformers/all-mpnet-base-v2`.
* --task can be one of: `mnli`, `rte`, `mrpc`, `sst5`, `dbpedia_14`, `nq`, `xsum`, `hellaswag`, `geoquery`.

To evaluate more embedding models, simply add the function `retrieval_xx`, and mapping entries in `RETRIEVAL_FUNCS` and `FORMAT_FUNCS` in `inference.py`.

### Evaluation
GeoQuery
```
python eval_geoquery.py --output_dir {output_dir}
```
other tasks
```
python eval.py --output_dir {output_dir} --task {task}
```
Note that the evaluation for MultiWoz is conducted with inference, so there is no need for separate running.

