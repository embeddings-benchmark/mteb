import mteb
import os
from tqdm import tqdm
import wandb
import json
import time


def is_run_available(model_name, model_revision):
    api = wandb.Api()
    runs = api.runs('Chembedding - Benchmarking')
    for run in runs:
        if run.name == model_name and run.config['revision'] == model_revision and run.state == "finished":
            return True
    return False


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def json_parser(data):
    task_name = data["task_name"]
    output = {}
    if task_name.endswith("PC"):
        output["PairClassification (Max F1)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Classification"):
        output["Classification (Accuracy)"] = data["scores"]["test"][0]["main_score"]
    elif "BitextMining" in task_name or task_name.endswith("BM"):
        output["Bitext Mining (F1)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Retrieval"):
        output["Retrieval (NDCG@10)"] = data["scores"]["test"][0]["main_score"]
    elif task_name.endswith("Clustering"):
        output["Clustering (V Measure)"] = data["scores"]["test"][0]["main_score"]
    return output


if __name__ == "__main__":
    now = time.time()

    models = {"google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
              "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
              "nomic-ai/nomic-bert-2048": "no_revision_available",
              "intfloat/e5-small": "e272f3049e853b47cb5ca3952268c6662abda68f",
              "intfloat/e5-base": "b533fe4636f4a2507c08ddab40644d20b0006d6a",
              "intfloat/e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
              "intfloat/e5-small-v2": "dca8b1a9dae0d4575df2bf423a5edb485a431236",
              "intfloat/e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
              "intfloat/e5-large-v2": "b322e09026e4ea05f42beadf4d661fb4e101d311",
              "intfloat/multilingual-e5-small": "e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
              "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
              "intfloat/multilingual-e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
              "nomic-ai/nomic-embed-text-v1": "0759316f275aa0cb93a5b830973843ca66babcf5",
              "nomic-ai/nomic-embed-text-v1.5": "b0753ae76394dd36bcfb912a46018088bca48be0",
              "recobo/chemical-bert-uncased": "498698d28fcf7ce5954852a0444c864bdf232b64",
              "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
              "BAAI/bge-small-en": "2275a7bdee235e9b4f01fa73aa60d3311983cfea",
              "BAAI/bge-base-en": "b737bf5dcc6ee8bdc530531266b4804a5d77b5d8",
              "BAAI/bge-large-en": "abe7d9d814b775ca171121fb03f394dc42974275",
              "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
              "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
              "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
              "all-mpnet-base-v2": "84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
              "multi-qa-mpnet-base-dot-v1": "3af7c6da5b3e1bea796ef6c97fe237538cbe6e7f",
              "all-MiniLM-L12-v2": "a05860a77cef7b37e0048a7864658139bc18a854",
              "all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
              "m3rg-iitd/matscibert": "ced9d8f5f208712c4a90f98a246fe32155b29995",
              "text-embedding-ada-002": "1",
              "text-embedding-3-small": "1",
              "text-embedding-3-large": "1",
              "amazon-titan-embed-text-v1": "1",
              "amazon-titan-embed-text-v2": "1",
              "cohere-embed-english-v3": "1",
              "cohere-embed-multilingual-v3": "1"
              }

    all_tasks = [
        # Pair Classification
        "CoconutSmiles2NamePC",
        "PubChemAIParagraphsParaphrasePC",
        "PubChemAISentenceParaphrasePC",
        "PubChemSMILESCanonDescPC",
        "PubChemSMILESCanonTitlePC",
        "PubChemSMILESIsoDescPC",
        "PubChemSMILESIsoTitlePC",
        "PubChemSynonymPC",
        "PubChemWikiParagraphsPC",
        # Classification
        "WikipediaEasy2GeneExpressionVsMetallurgyClassification",
        "WikipediaEasy2GreenhouseVsEnantiopureClassification",
        "WikipediaEasy2SolidStateVsColloidalClassification",
        "WikipediaEasy2SpecialClassification",
        "WikipediaEasy5Classification",
        "WikipediaEasy10Classification",
        "WikipediaEZ2Classification",
        "WikipediaEZ10Classification",
        "WikipediaHard2BioluminescenceVsLuminescenceClassification",
        "WikipediaHard2IsotopesVsFissionProductsNuclearFissionClassification",
        "WikipediaHard2SaltsVsSemiconductorMaterialsClassification",
        "WikipediaMedium2BioluminescenceVsNeurochemistryClassification",
        "WikipediaMedium2ComputationalVsSpectroscopistsClassification",
        "WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification",
        "WikipediaMedium5Classification",
        "SDSEyeProtectionClassification",
        "SDSGlovesClassification",
        # Retrieval
        "ChemNQRetrieval",
        "ChemHotpotQARetrieval",
        # Bitext Mining
        "CoconutSmiles2NameBitextMining1",
        "CoconutSmiles2NameBitextMining2",
        "PubChemSMILESISoTitleBM",
        "PubChemSMILESCanonTitleBM",
        "PubChemSMILESISoDescBM",
        "PubChemSMILESCanonDescBM",
        # Clustering
        "WikipediaEasy10Clustering",
        "WikipediaMedium5Clustering"
    ]

    tasks = mteb.get_tasks(tasks=all_tasks)

    for model_full_name, model_rev in tqdm(models.items()):
        if "/" in model_full_name:
            model_name = model_full_name.split("/")[1]
        else:
            model_name = model_full_name

        if is_run_available(model_name, model_rev):
            print(f"Skipping {model_name} - {model_rev}")
            continue

        try:
            wandb.init(project='Chembedding - Benchmarking', name=model_name,
                    config={"revision": model_rev})
            model = mteb.get_model(model_full_name)
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(model, output_folder="chem_results",
                        overwrite_results=False)
        except Exception as e:
            print(f"Error Evaluating Model {model_name}: {e}")
            wandb.finish()
            continue

        for task_name in tqdm(all_tasks):
            data = read_json(os.path.join(
                "chem_results",
                model_full_name.replace("/", "__"),
                model_rev,
                task_name + '.json',
            ))
            output = json_parser(data)
            wandb.log(output)

            for metric, score in output.items():
                table = wandb.Table(data=[[metric, score]],
                                    columns=["Metric", "Score"])
                bar_plot = wandb.plot.bar(
                    table, "Metric", "Score", title=f"{task_name} Performance")
                wandb.log({f"{task_name}_bar_plot": bar_plot})

        wandb.finish()

    elapsed = time.time() - now

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")