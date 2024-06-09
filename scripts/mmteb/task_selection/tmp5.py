import numpy as np


def test_significance(dist1, dist2, n_samples=10_000):
    """Test if two distributions are significantly different using a bootstrapping method

    # sanity check:
    # should be significant
    dist1=np.random.normal(1, 1, 10)
    dist2=np.random.normal(4, 1, 10)
    n_samples=10_000
    test_significance(dist1, dist2, n_samples)
    # should not be significant
    test_significance(np.random.normal(1, 1, 10), np.random.normal(1, 1, 10), n_samples=10_000)
    """
    # Compute the difference between the two distributions
    diff = abs(np.mean(dist1) - np.mean(dist2))

    # Concatenate the two distributions
    concat = np.concatenate([dist1, dist2])

    # Initialize the bootstrap samples
    bootstrap_diff = np.zeros(n_samples)

    # Generate the bootstrap samples
    for i in range(n_samples):
        # Generate a bootstrap sample
        bootstrap_sample = np.random.choice(concat, size=len(concat), replace=True)

        # Compute the difference between the two bootstrap samples
        bootstrap_diff[i] = abs(
            np.mean(bootstrap_sample[: len(dist1)])
            - np.mean(bootstrap_sample[len(dist1) :])
        )

    # Compute the p-value
    # I.e. what is the probability of observing a difference as extreme as the one we observed
    # given that the null hypothesis is true (i.e. the two distributions are the same)
    p_value = np.mean(bootstrap_diff >= diff)
    print(f"The p-value is: {p_value}")

    return p_value


def compute_significant_rank(scores: dict, threshold=0.05):
    """Compute significant rank for models.

    Example:
        ```
        scores = {"model1": np.random.normal(1, 1, 10) # 1 and 2 are similar
                "model2": np.random.normal(1.1, 1, 10)
                "model3": np.random.normal(5, 1, 10) # 3 is much better
        ranks = ranks(compute_significant_rank)
        print(ranks)
        # {
        "models": ["model3", "model2", "model1"],
        # "significant rank": [1, 2, 2],
        # "rank": [1, 2, 3],
        # }
        ```
    """
    ranks = {}

    mean_scores = [(m, np.mean(s)) for m, s in scores.items()]
    mean_scores = sorted(mean_scores, key=lambda x: -x[1])  # higher is first

    ranks["models"], _ = zip(*mean_scores)
    ranks["models"] = list(ranks["models"])
    ranks["rank"] = list(range(1, len(ranks["models"]) + 1))

    pairs = [
        (ranks["models"][i], ranks["models"][i + 1])
        for i in range(len(mean_scores) - 1)
    ]

    rank = 1
    ranks["significant rank"] = [rank]  # first model always get rank 1
    for p1, p2 in pairs:
        # test if the two models are significantly different
        p_value = test_significance(scores[p1], scores[p2])

        if p_value < threshold:
            rank += 1

        ranks["significant rank"].append(rank)

    return ranks


scores = {
    "model1": np.random.normal(1, 1, 10),  # 1 and 2 are similar
    "model2": np.random.normal(1.1, 1, 10),
    "model3": np.random.normal(5, 1, 10),  # 3 is much better
}
ranks = compute_significant_rank(scores)
assert ranks == {
    "models": ["model3", "model2", "model1"],
    "rank": [1, 2, 3],
    "significant rank": [1, 2, 2],
}

# we can then to spearman(ranks["significant rank"], ranks_from_another_task["significant rank"])
