import mteb
from mteb.tasks.Audio.Clustering.eng.VoiceGender import VoiceGenderClustering
from mteb.tasks.Audio.Clustering.eng.VoiceEmotions import CREMADEmotionClustering

# model_name = "microsoft/wavlm-base"
model_name = "Qwen/Qwen2-Audio-7B"
model = mteb.get_model(model_name)
print(f"Loaded model type: {type(model)}")
evaluation = mteb.MTEB(tasks=[CREMADEmotionClustering()])
cluster_algo = "Kmeans"
results = evaluation.run(model, output_folder=f"results_Emotions/{cluster_algo}/{model_name}", overwrite_results=True, cluster_algo=cluster_algo, limit=224)
print(results)