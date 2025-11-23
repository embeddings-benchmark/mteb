from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SynPerChatbotConvSAAnger(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAAnger",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Anger",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger",
            "revision": "5cae68b7fc094cb2fa6890a464e4d836e8107f5e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASatisfaction(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASatisfaction",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Satisfaction",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction",
            "revision": "50fd9d5d09edd53af89af765636be5db6f983f0e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAFriendship(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAFriendship",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Friendship",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship",
            "revision": "9dae119101e9b4e9bb40d5b9d29ffd7a621f9942",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAFear(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAFear",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Fear",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear",
            "revision": "3c22f7e6bf4e366c86d69293c9164bf9e9d80aac",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAJealousy(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAJealousy",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Jealousy",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy",
            "revision": "0d5104ecaa109d2448afe1f40dbf860f5e4458a8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASurprise(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASurprise",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Surprise",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise",
            "revision": "62dad66fc2837b0ac5e5175fe7c265d2d502a386",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSALove(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSALove",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Love",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love",
            "revision": "0e000b2f73e9bb74ec8fc6da10011c52725b8469",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSASadness(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSASadness",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Sadness",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness",
            "revision": "e9c678325565a5e4dadc43fd6693a8ccff1dd6b2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAHappiness(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAHappiness",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Happiness",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness",
            "revision": "e60893b7a8d01c9b8c12fadfe8f0a06e9d548a63",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAToneChatbotClassification",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-tone-chatbot-classification",
            "revision": "1f403cfadb85004fbf7e2480334fffc4c999b4ab",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotConvSAToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotConvSAToneUserClassification",
        description="Synthetic Persian Chatbot Conversational Sentiment Analysis Tone User",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/chatbot-conversational-sentiment-analysis-tone-user-classification",
            "revision": "dd0f76661bef69819cc38c8a455b10af86ac3571",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotSatisfactionLevelClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotSatisfactionLevelClassification",
        description="Synthetic Persian Chatbot Satisfaction Level Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-satisfaction-level-classification",
            "revision": "e72db473602d750f1bcdc9f0436e1e3b967e088f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotRAGToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGToneChatbotClassification",
        description="Synthetic Persian Chatbot RAG Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification",
            "revision": "76f15a203fc13bd98a8f0fdddab1b68c28d7d674",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotRAGToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGToneUserClassification",
        description="Synthetic Persian Chatbot RAG Tone User Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-tone-user-classification",
            "revision": "f1f6ad83bb135dc94fbf1ca05c3ba164f5619369",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotToneChatbotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotToneChatbotClassification",
        description="Synthetic Persian Chatbot Tone Chatbot Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-tone-chatbot-classification",
            "revision": "a5a739a036fa7bb8ae0be91bc081fdd260d4bdab",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerChatbotToneUserClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerChatbotToneUserClassification",
        description="Synthetic Persian Chatbot Tone User Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-tone-user-classification",
            "revision": "780d629437f7be127c6b287a61776372f9f333b9",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class SynPerTextToneClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerTextToneClassification",
        description="Persian Text Tone",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-text-tone-classification",
            "revision": "7144f4c6bdd77911df0dfc5a8bd44dba17e27e3a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
        superseded_by="SynPerTextToneClassification.v2",
    )
    samples_per_label = 32


class SynPerTextToneClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerTextToneClassification.v2",
        description="Persian Text Tone This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://mcinext.com/",
        dataset={
            "path": "mteb/syn_per_text_tone",
            "revision": "0ed7459db7e905714dc02cbe25b4eac55e91021e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
        adapted_from=["SynPerTextToneClassification"],
        superseded_by="SynPerTextToneClassification.v3",
    )
    samples_per_label = 32


class SynPerTextToneClassificationV3(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SynPerTextToneClassification.v3",
        description="This version of the Persian text tone classification dataset is an improved version of its predecessors. It excludes several classes identified as having low-quality data, leading to a more reliable benchmark.",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/synthetic-persian-text-tone-classification-v3",
            "revision": "ff6d88107a89abeb10aa28751b31d78831d7d503",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
        adapted_from=["SynPerTextToneClassification"],
    )
    samples_per_label = 32


class SIDClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SIDClassification",
        description="SID Classification",
        reference="https://mcinext.com/",
        dataset={
            "path": "MCINext/sid-classification",
            "revision": "29bed651bb980395f5aa473607154d93226945e1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        superseded_by="SIDClassification.v2",
    )
    samples_per_label = 32


class SIDClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SIDClassification.v2",
        description="SID Classification This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://mcinext.com/",
        dataset={
            "path": "mteb/sid",
            "revision": "8234b2081bd9ca33bdbc7bf68f5f9540fe3fd480",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["SIDClassification"],
    )
    samples_per_label = 32


class DeepSentiPers(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DeepSentiPers",
        description="Persian Sentiment Analysis Dataset",
        reference="https://github.com/JoyeBright/DeepSentiPers",
        dataset={
            "path": "PartAI/DeepSentiPers",
            "revision": "ee4f09f404051761cfe14d68127532c82de41cb3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        superseded_by="DeepSentiPers.v2",
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("review", "text")


class DeepSentiPersV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DeepSentiPers.v2",
        description="Persian Sentiment Analysis Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/JoyeBright/DeepSentiPers",
        dataset={
            "path": "mteb/deep_senti_pers",
            "revision": "8d60d8315ac650ef0af32d68c4f92916ffc5cfb8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["DeepSentiPers"],
    )
    samples_per_label = 32


class PersianTextEmotion(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersianTextEmotion",
        description="Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
        reference="https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion",
        dataset={
            "path": "SeyedAli/Persian-Text-Emotion",
            "revision": "518fcd2c8b89917c7696770672688217a2eabf88",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        superseded_by="PersianTextEmotion.v2",
    )
    samples_per_label = 32


class PersianTextEmotionV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersianTextEmotion.v2",
        description="Emotion is a Persian dataset with six basic emotions: anger, fear, joy, love, sadness, and surprise. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/SeyedAli/Persian-Text-Emotion",
        dataset={
            "path": "mteb/persian_text_emotion",
            "revision": "a45594021eca1d1577296edc030d972a92ff26b3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["PersianTextEmotion"],
    )
    samples_per_label = 32


class SentimentDKSF(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentDKSF",
        description="The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis.",
        reference="https://github.com/hezarai/hezar",
        dataset={
            "path": "hezarai/sentiment-dksf",
            "revision": "b4d5a8dd501db610b5ad89e9aa13f863b842b395",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        superseded_by="SentimentDKSF.v2",
    )
    samples_per_label = 32


class SentimentDKSFV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentimentDKSF.v2",
        description="The Sentiment DKSF (Digikala/Snappfood comments) is a dataset for sentiment analysis. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://github.com/hezarai/hezar",
        dataset={
            "path": "mteb/sentiment_dksf",
            "revision": "05129fb229c8f68267d112cffa655f1312ec6575",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["SentimentDKSF"],
    )
    samples_per_label = 32


class NLPTwitterAnalysisClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NLPTwitterAnalysisClassification",
        description="Twitter Analysis Classification",
        reference="https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main",
        dataset={
            "path": "hamedhf/nlp_twitter_analysis",
            "revision": "4ceb1312583fd2c7c73ad2d550b726124dcd39a0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        superseded_by="NLPTwitterAnalysisClassification.v2",
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("tweet", "text")


class NLPTwitterAnalysisClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NLPTwitterAnalysisClassification.v2",
        description="Twitter Analysis Classification This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main",
        dataset={
            "path": "mteb/nlp_twitter_analysis",
            "revision": "41d85185019495609522fece20e93d11ab705301",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        adapted_from=["NLPTwitterAnalysisClassification"],
    )
    samples_per_label = 32


class DigikalamagClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DigikalamagClassification",
        description="A total of 8,515 articles scraped from Digikala Online Magazine. This dataset includes seven different classes.",
        reference="https://hooshvare.github.io/docs/datasets/tc",
        dataset={
            "path": "mteb/DigikalamagClassification",
            "revision": "1425e8f2c0e68c32dbabfabe818fcc73e24079bb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32


class FaIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FaIntentClassification",
        description="Questions in 4 different categories that a user might ask their voice assistant to do",
        reference="https://github.com/HalflingWizard/FA-Intent-Classification-and-Slot-Filling",
        dataset={
            "path": "MCINext/FaIntent",
            "revision": "fc380690afbee9dba4dc618ef852285fa26f1d51",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2021-09-01", "2021-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("words", "text")
        self.dataset = self.dataset.rename_column("intent_label", "label")


class StyleClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="StyleClassification",
        description="A dataset containing formal and informal sentences in Persian for style classification.",
        reference="https://huggingface.co/datasets/MCINext/style-classification",
        dataset={
            "path": "MCINext/style-classification",
            "revision": "41a0848f718a28b9a6333b2be47b6dc93d5c1803",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
    )
    samples_per_label = 32

    def dataset_transform(self):
        mapping = {"formal": 1, "informal": 0}
        self.dataset = self.dataset.map(
            lambda example: {"label": mapping[example["label"]]}
        )


class PerShopDomainClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PerShopDomainClassification",
        description="PerSHOP - A Persian dataset for shopping dialogue systems modeling",
        reference="https://github.com/keyvanmahmoudi/PerSHOP",
        dataset={
            "path": "MCINext/pershop-classification",
            "revision": "05027cfce1d20ab7c9f4755b064ea6958cdee96e",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2023-09-01", "2024-01-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""@article{mahmoudi2024pershop,
  author = {Mahmoudi, Keyvan and Faili, Heshaam},
  journal = {arXiv preprint arXiv:2401.00811},
  title = {PerSHOP--A Persian dataset for shopping dialogue systems modeling},
  year = {2024},
}""",
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("domain", "label")


class PerShopIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PerShopIntentClassification",
        description="PerSHOP - A Persian dataset for shopping dialogue systems modeling",
        reference="https://github.com/keyvanmahmoudi/PerSHOP",
        dataset={
            "path": "MCINext/pershop-classification",
            "revision": "05027cfce1d20ab7c9f4755b064ea6958cdee96e",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="accuracy",
        date=("2023-09-01", "2024-01-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""@article{mahmoudi2024pershop,
  author = {Mahmoudi, Keyvan and Faili, Heshaam},
  journal = {arXiv preprint arXiv:2401.00811},
  title = {PerSHOP--A Persian dataset for shopping dialogue systems modeling},
  year = {2024},
}""",
    )
    samples_per_label = 32

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("Intents & Actions", "label")
