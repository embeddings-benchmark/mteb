# create a normal distribution and transform it using a inverse logit function
import numpy as np

# Creating synthetic data
np.random.seed(42)

n_samples = 10_000
X = np.random.normal(size=n_samples, loc=1.1, scale=2)


# plot
import matplotlib.pyplot as plt

plt.hist(X, bins=30)
plt.show()

# inverse logit function
def invlogit(x):
    return 1 / (1 + np.exp(-x))

# plot the transformed data
X_invlogit = invlogit(X)
plt.hist(X_invlogit, bins=30)