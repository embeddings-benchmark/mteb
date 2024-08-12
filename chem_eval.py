import mteb
from tqdm import tqdm

models = ["allenai/scibert_scivocab_uncased",
          "google-bert/bert-base-uncased",
          "intfloat/multilingual-e5-small",
          "intfloat/multilingual-e5-base",
          "intfloat/multilingual-e5-large",
          "nomic-ai/nomic-embed-text-v1.5",
          "nomic-ai/nomic-embed-text-v1",
          "nomic-ai/nomic-bert-2048"
          ]

tasks = mteb.get_tasks(tasks=["PubChemAIParagraphsParaphrasePC",
                              "PubChemAISentenceParaphrasePC",
                              "PubChemSynonymPC",
                              "PubChemSMILESIsoTitlePC",
                              "PubChemSMILESIsoDescPC",
                              "PubChemSMILESCanonTitlePC",
                              "PubChemSMILESCanonDescPC",
                              "PubChemWikiParagraphsPC",
                              "ChemNQRetrieval",
                              "ChemHotpotQARetrieval"
                              ])

for model_name in tqdm(models):
    model = mteb.get_model(model_name)
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder="chem_results")
