from .Evaluator import Evaluator
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
        knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        X_train = np.asarray(model.encode(self.sentences_train))
        X_test = np.asarray(model.encode(self.sentences_test))

        knn.fit(X_train, self.y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return {"accuracy": accuracy}
