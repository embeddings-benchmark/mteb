"""Stratification utilities for multi-label data sets

The following code is a copy of
https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/model_selection/iterative_stratification.py
from the amazing scikit-multilearn library. Please try it out: https://github.com/scikit-multilearn/scikit-multilearn
and cite the references above:

Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158.
http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017.
http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html

Bibtex:

.. code-block:: bibtex

    @article{sechidis2011stratification,
      title={On the stratification of multi-label data},
      author={Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis},
      journal={Machine Learning and Knowledge Discovery in Databases},
      pages={145--158},
      year={2011},
      publisher={Springer}
    }

    @InProceedings{pmlr-v74-szymański17a,
      title =    {A Network Perspective on Stratification of Multi-Label Data},
      author =   {Piotr Szymański and Tomasz Kajdanowicz},
      booktitle =    {Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications},
      pages =    {22--35},
      year =     {2017},
      editor =   {Luís Torgo and Bartosz Krawczyk and Paula Branco and Nuno Moniz},
      volume =   {74},
      series =   {Proceedings of Machine Learning Research},
      address =      {ECML-PKDD, Skopje, Macedonia},
      publisher =    {PMLR},
    }
"""

import itertools
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state


def _iterative_train_test_split(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    test_size: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively stratified train/test split

    Slighltly modified from:
        https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/model_selection/iterative_stratification.py

    Args:
        X: np.ndarray, shape (n_samples, n_features) input samples
        y: np.ndarray, shape (n_samples, n_labels) multi-label binary matrix
        test_size: float, [0,1] the proportion of the dataset to include in the test split, the rest will be put in the train set
        random_state: random state seed

    Returns:
        train_indexes, test_indexes: np.ndarray, np.ndarray, the indexes of the train and test samples
    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
        random_state=random_state,
    )
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes, test_indexes


def _fold_tie_break(
    desired_samples_per_fold: np.ndarray,
    M: np.ndarray,  # noqa: N803
    random_state: np.random.RandomState,
):
    """Helper function to split a tie between folds with same desirability of a given sample

    Args:
        desired_samples_per_fold: np.array[Float], number of samples desired per fold
        M: np.array(int), List of folds between which to break the tie
        random_state: A numpy random state

    Returns:
        The selected fold index to put samples into
    """
    if len(M) == 1:
        return M[0]
    else:
        max_val = max(desired_samples_per_fold[M])
        m_prim = np.where(np.array(desired_samples_per_fold) == max_val)[0]
        m_prim = np.array([x for x in m_prim if x in M])
        return random_state.choice(m_prim, 1)[0]


def _get_most_desired_combination(samples_with_combination: dict):
    """Select the next most desired combination whose evidence should be split among folds

    Args:
        samples_with_combination : dict[Combination, list[int]], :code:`(n_combinations)`
            map from each label combination present in y to list of sample indexes that have this combination assigned

    Returns:
        Combination, the combination to split next
    """
    currently_chosen = None
    best_number_of_combinations, best_support_size = None, None

    for combination, evidence in samples_with_combination.items():
        number_of_combinations, support_size = (len(set(combination)), len(evidence))
        if support_size == 0:
            continue
        if currently_chosen is None or (
            best_number_of_combinations is not None
            and best_support_size is not None
            and best_number_of_combinations < number_of_combinations
            and best_support_size > support_size
        ):
            currently_chosen = combination
            best_number_of_combinations, best_support_size = (
                number_of_combinations,
                support_size,
            )

    return currently_chosen


class IterativeStratification(_BaseKFold):
    """Iteratively stratify a multi-label data set into folds

    Construct an iterative stratifier that splits the data set into folds trying to maintain balanced representation
    with respect to order-th label combinations.
    """

    def __init__(
        self,
        n_splits: int = 3,
        order: int = 1,
        sample_distribution_per_fold: list[float] | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the stratifier

        Args:
            n_splits: number of folds. Must be at least 2.
            order: order of label combinations to use in stratification. 1 means single labels, 2 means pairs of labels, etc.
            sample_distribution_per_fold: list[float], list of length n_splits with the proportion of samples to assign to each fold. If None, uniform distribution is used.
            shuffle: bool, whether to shuffle the data before splitting into batches. Note that the samples are always shuffled within each label combination.
            random_state: int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`. Used when `shuffle` is True.
        """
        self._rng_state = check_random_state(random_state)
        need_shuffle = shuffle or random_state is not None
        self.order = order
        super().__init__(
            n_splits,
            shuffle=need_shuffle,
            random_state=self._rng_state if need_shuffle else None,
        )

        if sample_distribution_per_fold:
            self.percentage_per_fold = sample_distribution_per_fold
        else:
            self.percentage_per_fold = [
                1 / float(self.n_splits) for _ in range(self.n_splits)
            ]

    def _prepare_stratification(
        self, y: np.ndarray
    ) -> tuple[
        list[list[int]],
        dict[int, bool],
        list[list[int]],
        list[list[Any]],
        dict[str, list[Any]],
        list[list[Any]],
    ]:
        """Prepares variables for performing stratification

        For the purpose of clarity, the type Combination denotes list[int], :code:`(self.order)` and represents a
        label combination of the order we want to preserve among folds in stratification. The total number of
        combinations present in :code:`(y)` will be denoted as :code:`(n_combinations)`.

        Arguments:
            y: output matrix or array of arrays (n_samples, n_labels)

        Returns:
            A tuple containing the following elements:
            - rows : list[list[int]], :code:`(n_samples, n_labels)` list of label indices assigned to each sample
            - rows_used : dict[int, bool], :code:`(n_samples)` boolean map from a given sample index to boolean value whether it has been already assigned to a fold or not
            - all_combinations :  list[Combination], :code:`(n_combinations)` list of all label combinations of order self.order present in y
            - per_row_combinations : list[Combination], :code:`(n_samples)` list of all label combinations of order self.order present in y per row
            - samples_with_combination : dict[Combination, list[int]], :code:`(n_combinations)` map from each label combination present in y to list of sample indexes that have this combination assigned
            - folds: list[list[int]] (n_splits) list of lists to be populated with samples
        """
        self.n_samples, self.n_labels = y.shape
        self.desired_samples_per_fold = np.array(
            [self.percentage_per_fold[i] * self.n_samples for i in range(self.n_splits)]
        )
        rows = sp.lil_matrix(y).rows
        rows_used = dict.fromkeys(range(self.n_samples), False)
        all_combinations = []
        per_row_combinations: list[list[Any]] = [[] for i in range(self.n_samples)]
        samples_with_combination: dict[str, list[Any]] = {}
        folds: list[list[int]] = [[] for _ in range(self.n_splits)]

        # for every row
        for sample_index, label_assignment in enumerate(rows):
            # for every n-th order label combination
            # register combination in maps and lists used later
            for combination in itertools.combinations_with_replacement(
                label_assignment, self.order
            ):
                if combination not in samples_with_combination:
                    samples_with_combination[combination] = []

                samples_with_combination[combination].append(sample_index)
                all_combinations.append(combination)
                per_row_combinations[sample_index].append(combination)

        self.desired_samples_per_combination_per_fold = {
            combination: np.array(
                [
                    len(evidence_for_combination) * self.percentage_per_fold[j]
                    for j in range(self.n_splits)
                ]
            )
            for combination, evidence_for_combination in samples_with_combination.items()
        }
        return (
            rows.tolist(),
            rows_used,
            [list(x) for x in set(all_combinations)],
            per_row_combinations,
            samples_with_combination,
            folds,
        )

    def _distribute_positive_evidence(
        self, rows_used, folds, samples_with_combination, per_row_combinations
    ):
        """Internal method to distribute evidence for labeled samples across folds

        For params, see documentation of :code:`self._prepare_stratification`. Does not return anything,
        modifies params.
        """
        l = _get_most_desired_combination(samples_with_combination)
        while l is not None:
            while len(samples_with_combination[l]) > 0:
                row = samples_with_combination[l].pop()
                if rows_used[row]:
                    continue

                max_val = max(self.desired_samples_per_combination_per_fold[l])
                m = np.where(
                    np.array(self.desired_samples_per_combination_per_fold[l])
                    == max_val
                )[0]
                m = _fold_tie_break(
                    self.desired_samples_per_combination_per_fold[l], m, self._rng_state
                )
                folds[m].append(row)
                rows_used[row] = True
                for i in per_row_combinations[row]:
                    if row in samples_with_combination[i]:
                        samples_with_combination[i].remove(row)
                    self.desired_samples_per_combination_per_fold[i][m] -= 1
                self.desired_samples_per_fold[m] -= 1

            l = _get_most_desired_combination(samples_with_combination)

    def _distribute_negative_evidence(self, rows_used, folds):
        """Internal method to distribute evidence for unlabeled samples across folds

        For params, see documentation of :code:`self._prepare_stratification`. Does not return anything,
        modifies params.
        """
        available_samples = [i for i, v in rows_used.items() if not v]
        samples_left = len(available_samples)

        while samples_left > 0:
            row = available_samples.pop()
            rows_used[row] = True
            samples_left -= 1
            fold_selected = self._rng_state.choice(
                np.where(self.desired_samples_per_fold > 0)[0], 1
            )[0]
            self.desired_samples_per_fold[fold_selected] -= 1
            folds[fold_selected].append(row)

    def _iter_test_indices(self, X, y=None, groups=None):  # noqa: N803
        """Internal method for providing scikit-learn's split with folds

        Args:
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.
                Note that providing ``y`` is sufficient to generate the splits and
                hence ``np.zeros(n_samples)`` may be used as a placeholder for
                ``X`` instead of actual training data.
            y : array-like, shape (n_samples,)
                The target variable for supervised learning problems.
                Stratification is done based on the y labels.
            groups : object
                Always ignored, exists for compatibility.

        Yields:
            fold : list[int]
                indexes of test samples for a given fold, yielded for each of the folds
        """
        (
            rows,
            rows_used,
            all_combinations,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = self._prepare_stratification(y)

        self._distribute_positive_evidence(
            rows_used, folds, samples_with_combination, per_row_combinations
        )
        self._distribute_negative_evidence(rows_used, folds)

        yield from folds
