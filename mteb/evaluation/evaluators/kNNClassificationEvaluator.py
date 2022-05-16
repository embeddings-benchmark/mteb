from .Evaluator import Evaluator
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


class kNNClassificationEvaluator(Evaluator):
    def __init__(self, sentences_train, y_train, sentences_test, y_test, k=3):
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        seed = 28042000
        random.seed(seed)
        np.random.seed(seed)

        self.k = k

    def __call__(self, model):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        for metric in ["cosine", "euclidean"]: #TODO: "dot"
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            X_train = np.asarray(model.encode(self.sentences_train))
            X_test = np.asarray(model.encode(self.sentences_test))
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        return scores
