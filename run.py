import mteb
from mteb.tasks.Audio.Clustering.eng.VoiceGender import VoiceGenderClustering

# model_name = "facebook/wav2vec2-large-xlsr-53"
model_name = "microsoft/wavlm-base-sd"

model = mteb.get_model(model_name, revision="main")
print(f"Loaded model type: {type(model)}")
# tasks = mteb.get_tasks(tasks=["VoiceGenderClustering"])
evaluation = mteb.MTEB(tasks=[VoiceGenderClustering()])
# evaluation = mteb.MTEB(tasks=task s)

results = evaluation.run(model, output_folder=f"results/{model_name}")
print(results)
