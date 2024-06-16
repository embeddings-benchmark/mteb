import json
# from pathlib import Path
from datasets import load_dataset,get_dataset_split_names,get_dataset_infos
import mteb


def extract_bibtex_to_file(tasks: list[mteb.AbsTask]) -> None:
    """Parse the task and extract bibtex.
    :param tasks:
        List of tasks.
    """
    # titles = []
    # bibtex = []
    c = dict()
    for task in tasks:
        data = task.metadata.dataset
        if not task.metadata.is_filled():
            # c.append(task.metadata.name)
            if task.metadata.avg_character_length is not None:
                c[task.metadata.name] = [task.metadata.type,task.metadata.eval_langs]

        # if data.get('trust_remote_code',None): c+=1
        # continue
        # try:
        #     # if not data['revision']: 
        #     dataset = get_dataset_infos(**data)
        #     # print(dataset)
        #     # else:
        #     #     dataset = load_dataset(data['path'], revision=data['revision'])
        # except Exception as e:
        #     # if "trust_remote_code=True" in str(e):
        #     print(str(e), data)
        #     break
    # print(c)
    with open("tasks_verified.json",'w') as f:
        json.dump(c,f,indent=4)



def main():
    # dataset = load_dataset("nilc-nlp/assin2")
    # print(get_dataset_split_names("shunk031/JGLUE",'MARC-ja'))
    tasks = mteb.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.metadata.name)
    extract_bibtex_to_file(tasks)
    # print(len(tasks)) # 505 dont have them.



if __name__ == "__main__":
    main()  #AmazonCounterfactualClassification

# import mteb
# dataset = load_dataset("mteb/amazon_counterfactual",trust_remote_code=False)
# from sentence_transformers import SentenceTransformer

# # Define the sentence-transformers model name
# model_name = "average_word_embeddings_komninos"
# # or directly from huggingface:
# # model_name = "sentence-transformers/all-MiniLM-L6-v2"

# model = SentenceTransformer(model_name)
# tasks = mteb.get_tasks(tasks=["AmazonCounterfactualClassification"])
# evaluation = mteb.MTEB(tasks=tasks)
# results = evaluation.run(model, output_folder=f"results/{model_name}")


