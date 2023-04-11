import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class kNNClassificationEvaluator(Evaluator):
    def __init__(self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean"]:  # TODO: "dot"
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            # if binary classification
            if len(np.unique(self.y_train)) == 2:
                ap = average_precision_score(self.y_test, y_pred)
                scores["ap_" + metric] = ap
                max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = max_ap
        return scores, test_cache


class kNNClassificationEvaluatorPytorch(Evaluator):
    def __init__(self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]

        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean", "dot"]:
            if metric == "cosine":
                distances = 1 - self._cos_sim(X_test, X_train)
            elif metric == "euclidean":
                distances = self._euclidean_dist(X_test, X_train)
            elif metric == "dot":
                distances = -self._dot_score(X_test, X_train)
            neigh_indices = torch.topk(distances, k=self.k, dim=1, largest=False).indices
            y_train = torch.tensor(self.y_train)
            y_pred = torch.mode(y_train[neigh_indices], dim=1).values  # TODO: case where there is no majority
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            # if binary classification
            if len(np.unique(self.y_train)) == 2:
                ap = average_precision_score(self.y_test, y_pred)
                scores["ap_" + metric] = ap
                max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        if len(np.unique(self.y_train)) == 2:
            scores["ap"] = max_ap
        return scores, test_cache

    @staticmethod
    def _cos_sim(a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euclidean_dist(a: Tensor, b: Tensor):
        """
        Computes the euclidean distance euclidean_dist(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = euclidean_dist(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.cdist(a, b, p=2)

    @staticmethod
    def _dot_score(a: Tensor, b: Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))


class logRegClassificationEvaluator(Evaluator):
    def __init__(
        self, sentences_train, y_train, sentences_test, y_test, max_iter=100, batch_size=32, limit=None, **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.max_iter = max_iter
        self.batch_size = batch_size

    def __call__(self, model, test_cache=None):
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        logger.info(f"Encoding {len(self.sentences_train)} training sentences...")
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        logger.info(f"Encoding {len(self.sentences_test)} test sentences...")
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        scores["accuracy"] = accuracy
        scores["f1"] = f1

        # if binary classification
        if len(np.unique(self.y_train)) == 2:
            ap = average_precision_score(self.y_test, y_pred)
            scores["ap"] = ap

        return scores, test_cache
