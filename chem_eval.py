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
    return output


if __name__ == "__main__":
    now = time.time()

    models = {"google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
              "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
              "nomic-ai/nomic-bert-2048": "no_revision_available",
              "intfloat/multilingual-e5-small": "e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
              "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
              "intfloat/multilingual-e5-large": "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
              "nomic-ai/nomic-embed-text-v1": "0759316f275aa0cb93a5b830973843ca66babcf5",
              "nomic-ai/nomic-embed-text-v1.5": "b0753ae76394dd36bcfb912a46018088bca48be0",
              "recobo/chemical-bert-uncased": "498698d28fcf7ce5954852a0444c864bdf232b64",
              "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
              "all-mpnet-base-v2": "84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
              "multi-qa-mpnet-base-dot-v1": "3af7c6da5b3e1bea796ef6c97fe237538cbe6e7f",
              "all-MiniLM-L12-v2": "a05860a77cef7b37e0048a7864658139bc18a854",
              "all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
              "m3rg-iitd/matscibert": "ced9d8f5f208712c4a90f98a246fe32155b29995",
              "text-embedding-ada-002": "1",
              "text-embedding-3-small": "1",
              "text-embedding-3-large": "1",
              }

    all_tasks = [
        "CoconutSmiles2NamePC",
        "PubChemAIParagraphsParaphrasePC",
        "PubChemAISentenceParaphrasePC",
        "PubChemSMILESCanonDescPC",
        "PubChemSMILESCanonTitlePC",
        "PubChemSMILESIsoDescPC",
        "PubChemSMILESIsoTitlePC",
        "PubChemSynonymPC",
        "PubChemWikiParagraphsPC",

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

        "ChemNQRetrieval",
        "ChemHotpotQARetrieval",

        "CoconutSmiles2NameBitextMining1",
        "CoconutSmiles2NameBitextMining2",
        "PubChemSMILESISoTitleBM",
        "PubChemSMILESCanonTitleBM",
        "PubChemSMILESISoDescBM",
        "PubChemSMILESCanonDescBM"
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

        wandb.init(project='Chembedding - Benchmarking', name=model_name,
                   config={"revision": model_rev})
        model = mteb.get_model(model_full_name)
        evaluation = mteb.MTEB(tasks=tasks)
        evaluation.run(model, output_folder="chem_results",
                       overwrite_results=False)

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