import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

sys.path.append(str(Path(__file__).parent.parent))
from get_table_from_hf import get_leaderboard_df

results_path = Path(__file__).parent / ".." / "results.csv"
if not results_path.exists():
    get_leaderboard_df(str(results_path))
# columns = tasks, rows = embedding methods
data = pd.read_csv(str(results_path))
data = data.set_index("Model")
overall = data["Overall"]
data = data.drop(columns=["Overall"])


tasks = data.columns
np.random.seed(42)
tasks_to_keep = np.random.choice(tasks, 14, replace=False)

tasks_to_keep = list(tasks_to_keep)
task = "ArxivClusteringS2S"
n = 10
for i in range(n):
    tasks_to_keep.append(task)

data_to_keeep = data[tasks_to_keep]


test_scores = data_to_keeep.to_numpy() / 100
test_scores = np.clip(test_scores, 1e-6, 1 - 1e-6) #


n_subjects, n_tests = test_scores.shape

# Create the PyMC model
with pm.Model() as model:
    population_g = pm.Normal("population_g", mu=0, sigma=10)
    g = pm.Normal("g", mu=population_g, sigma=1, shape=n_subjects)

    # # Priors for loadings
    # loadings = pm.Gamma("phi", alpha=2, beta=1, shape=n_tests)
    # Priors for loadings with a correlation structure
    chol, corr, stds = pm.LKJCholeskyCov('chol', n=n_tests, eta=2, sd_dist=pm.Exponential.dist(1.0))
    loadings_raw = pm.MvNormal("loadings_raw", mu=np.zeros(n_tests), chol=chol, shape=n_tests)
    loadings = pm.Deterministic('phi', pm.math.exp(loadings_raw))  # Ensuring positive loadings



    # Transformed g to be between 0 and 1
    g_transformed = pm.math.invlogit(g)  # type: ignore

    # Parameters for the Beta distribution
    alpha = g_transformed[:, None] * loadings[None, :]
    beta =  (1 - g_transformed[:, None]) * loadings[None, :] 

    # Likelihood for the narrow abilities (test scores)
    test_scores_obs = pm.Beta(
        "test_scores_obs", alpha=alpha, beta=beta, observed=test_scores
    )


pm.model_to_graphviz(model)


# Inference / Model fitting
with model:
    # Inference
    idata = pm.sample(1000, tune=1000, chains=4, cores=4, target_accept=0.8)

# Posterior analysis
# az.plot_trace(idata)
# az.summary(idata)


# get the q of a subject
g_posterior_samples = idata.posterior["g"]
g_estimates = np.mean(g_posterior_samples, axis=(0, 1))

data["g_estimates"] = g_estimates
data["Overall"] = overall
data[["g_estimates", "Overall"]]

# plot the estimates
import matplotlib.pyplot as plt

plt.scatter(data["Overall"], data["g_estimates"])
plt.xlabel("Overall Score")
plt.ylabel("Estimated g")
plt.title("Estimated g vs Overall Score (13 random + 4 duplicate tasks, 14 unique tasks in total)")
plt.show()


data["Overall_w_duplicates"] = data_to_keeep.to_numpy().mean(axis=1)
plt.scatter(data["Overall"], data["Overall_w_duplicates"])
plt.xlabel("Overall Score")
plt.ylabel("Overall Score with duplicates")
plt.title("Overall Score vs Overall Score with duplicates")


# create a new df
# g_posterior_samples.data.reshape(-1, n_subjects).shape

model_g_post = g_posterior_samples.data.reshape(-1, n_subjects)
model_g_post = pd.DataFrame(model_g_post, columns=data.index)

top_models = data["Overall"].sort_values(ascending=False).index[:5]


# Model 
import seaborn as sns
sns.violinplot(data=model_g_post[[*top_models]], native_scale=True)
plt.xticks(rotation=45)
plt.xlabel("Model")
plt.ylabel("g")





# sort
data = data.sort_values("g_estimates", ascending=False)

