"""
Experimental script to test the effect of downsampling on the results of the MTEB benchmark. Will not be submitted.
"""

from mteb import ArxivClusteringS2S

target_task = ArxivClusteringS2S
task = target_task()

task.load_data()


t = []
for i in range(len(task.dataset["test"]["sentences"])):
    t += list(
        zip(task.dataset["test"]["sentences"][i], task.dataset["test"]["labels"][i])
    )

# assert len(t) == len(set(t)) # this is false(!)

from sentence_transformers import SentenceTransformer

from mteb import MTEB

bench = MTEB(tasks=[task])

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

bench.run(model, output_folder="tmp_res/results_before", overwrite_results=False)

# downsample task
dataset = {"test": []}

for i in range(len(task.dataset["test"]["sentences"])):
    dataset["test"].append(
        {
            "sentences": task.dataset["test"]["sentences"][i][:10000],
            "labels": task.dataset["test"]["labels"][i][:10000],
        }
    )
downsampled_task = target_task()
downsampled_task.dataset = dataset
downsampled_task.data_loaded = True

bench = MTEB(tasks=[downsampled_task])
bench.run(model, output_folder="tmp_res/results_after_10k", overwrite_results=True)
