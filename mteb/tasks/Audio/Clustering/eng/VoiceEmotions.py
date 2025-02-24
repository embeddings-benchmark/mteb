from mteb.abstasks.Audio.AbsTaskAudioClustering import AbsTaskAudioClustering
from mteb.abstasks.TaskMetadata import TaskMetadata
import random
import datasets
import mteb
from mteb import MTEB


class CREMADEmotionClustering(AbsTaskAudioClustering):
    label_column_name: str = "label"

    metadata = TaskMetadata(
        name="CREMADEmotionClustering",
        description="Clustering audio recordings based on expressed emotions from the CREMA-D dataset.",
        reference="https://huggingface.co/datasets/AbstractTTS/CREMA-D",
        dataset={
            "path": "AbstractTTS/CREMA-D",
            "revision": "main",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["train"],  
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=("2014-01-01", "2024-12-31"),  
        domains=["Spoken"],
        task_subtypes=["Voice Emotion Clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
    )

    def dataset_transform(self):
        EMOTION_MAP = {"anger": 0,"happy": 1,"neutral": 2,"sad": 3,"fear": 4,"disgust": 5}
        splits = self.metadata.eval_splits
        ds = {}
        for split in splits:
            ds_split = self.dataset[split]
            audio = ds_split["audio"]
            labels = ds_split["major_emotion"]
            audio = [{"array": item["array"], "sampling_rate": item["sampling_rate"]} for item in audio]
            labels = [EMOTION_MAP.get(str(label).lower().strip(), -1) for label in labels]
            rng = random.Random(1111)
            data_pairs = list(zip(audio, labels))
            rng.shuffle(data_pairs)
            audio, labels = zip(*data_pairs)

            batch_size = 512
            audio_batched = [audio[i:i + batch_size] for i in range(0, len(audio), batch_size)]
            labels_batched = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]

            audio_batched = audio_batched[:4]
            labels_batched = labels_batched[:4]
            
            audio_batched = [item for batch in audio_batched for item in batch]
            labels_batched = [item for batch in labels_batched for item in batch]

            ds[split] = datasets.Dataset.from_dict({
                "audio": audio_batched,
                "label": labels_batched,
            })

        self.dataset = datasets.DatasetDict(ds)
        


if __name__ == "__main__":
    model_name = "microsoft/wavlm-base"
    model_name = "facebook/wav2vec2-base"
    model = mteb.get_model(model_name)
    print(f"Loaded model type: {type(model)}")
    evaluation = mteb.MTEB(tasks=[CREMADEmotionClustering()])
    cluster_algo = "Kmeans"
    results = evaluation.run(model, output_folder=f"results_Emotions/{cluster_algo}/{model_name}", overwrite_results=True, cluster_algo=cluster_algo, limit=224)
    print(results)





