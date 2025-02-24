from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata
import mteb
from mteb import MTEB



class VoiceGenderClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"
    metadata = TaskMetadata(
        name="VoiceGenderClustering",
        description="Clustering audio recordings based on gender (male vs female).",
        reference="https://huggingface.co/datasets/mmn3690/voice-gender-clustering",
        dataset={
            "path": "mmn3690/voice-gender-clustering",
            "revision": "main",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Voice Gender Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
    )

if __name__ == "__main__":
    #model_name = "microsoft/wavlm-base"
    model_name = "facebook/wav2vec2-base"
    model = mteb.get_model(model_name)
    print(f"Loaded model type: {type(model)}")
    evaluation = mteb.MTEB(tasks=[VoiceGenderClustering()])
    cluster_algo = "Kmeans"
    results = evaluation.run(model, output_folder=f"results/{cluster_algo}/{model_name}", overwrite_results=True, cluster_algo=cluster_algo)
    print(results)
    
    # from datasets import load_dataset
    # dataset = load_dataset("mmn3690/voice-gender-clustering", split="train")
    # print(dataset["label"]) 
