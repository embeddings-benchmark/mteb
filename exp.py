from mteb import ArxivClusteringP2P

task = ArxivClusteringP2P() 

task.load_data()


# checking that there are no duplicates
# for i in range(len(task.dataset["test"]["sentences"])):
#     print(
#         len(task.dataset["test"]["sentences"][i]),
#         len(set(task.dataset["test"]["sentences"][i])),
#     )

# t = list(zip(task.dataset["test"]["sentences"][9], task.dataset["test"]["labels"][9]))

# checking that there are no cross duplicates

t = []
for i in range(len(task.dataset["test"]["sentences"])):
    t += list(zip(task.dataset["test"]["sentences"][i], task.dataset["test"]["labels"][i]))
    
# assert len(t) == len(set(t)) # this is false

from sentence_transformers import SentenceTransformer
from mteb import MTEB

# bench = MTEB(tasks=[task])

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# bench.run(model, output_folder="tmp_res/results_before", overwrite_results=True)

# downsample task
dataset = {"test": []}

for i in range(len(task.dataset["test"]["sentences"])):
    # dataset["test"]["sentences"].append(task.dataset["test"]["sentences"][i][:1000])
    # dataset["test"]["labels"].append(task.dataset["test"]["labels"][i][:1000])
    dataset["test"].append(
        {
            "sentences": task.dataset["test"]["sentences"][i][:5000],
            "labels": task.dataset["test"]["labels"][i][:5000],
        }
    )
downsampled_task = ArxivClusteringP2P()
downsampled_task.dataset = dataset
downsampled_task.data_loaded = True

bench = MTEB(tasks=[downsampled_task])
bench.run(model, output_folder="tmp_res/results_after_5k", overwrite_results=True)