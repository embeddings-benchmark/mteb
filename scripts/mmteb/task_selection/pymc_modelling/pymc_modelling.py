"""For more on the description of this see the corresponding markdown


Requires:
    pip install pymc
    pip install graphviz
    (plus the graphviz library installed on your system, e.g. installed using `brew install graphviz`)
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from get_table_from_hf import get_leaderboard_df

results_path = Path(__file__).parent / ".." / "results.csv"
if not results_path.exists():
    get_leaderboard_df(str(results_path))
# columns = tasks, rows = embedding methods
data = pd.read_csv(str(results_path))
data = data.set_index("Model")
data = data.drop(columns=["Overall"])

task_to_predict = "AmazonPolarityClassification"

# Prepare the data for modelling
labels = [col for col in data.columns if col != task_to_predict]
# labels = labels[:10]
X_train, X_test, y_train, y_test = train_test_split(
    data[labels], data[[task_to_predict]], test_size=0.20, random_state=42
)

X_train = X_train.to_numpy() / 100
X_test = X_test.to_numpy() / 100
y_train = (y_train.to_numpy() / 100).flatten()
y_test = (y_test.to_numpy() / 100).flatten()


# Define the model
coords = {"coeffs": labels}
with pm.Model(coords=coords) as model:
    X = pm.MutableData("Task Performance (Observed)", X_train)
    y = pm.MutableData(
        "Task Performance (Unobserved)", y_train
    )  # task performance on unseen task (to be predicted)

    # Priors
    meta_beta = pm.Normal("meta beta", mu=0, sigma=10)
    meta_beta = 0
    beta = pm.Normal("beta", mu=meta_beta, sigma=10, dims="coeffs")
    intercept = pm.Normal("intercept", mu=meta_beta, sigma=10)

    # Linear combination of inputs
    mu_linear = pm.math.dot(X, beta) + intercept  # type: ignore

    # # Normal likelihood (for testing)
    # sigma = pm.HalfNormal("sigma", sigma=1)
    # y_obs = pm.Normal("y_obs", mu=mu_linear, sigma=sigma, observed=y)
    
    # Logit link function to ensure mu is between 0 and 1
    mu = pm.math.invlogit(mu_linear)  # type: ignore
    # Precision parameter (optional to estimate or set as a constant)
    phi = pm.Gamma("phi", alpha=2, beta=2)

    alpha = mu * phi
    beta = (1 - mu) * phi

    # # Likelihood
    y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta, observed=y)

    # # Normal likelihood (for testing)
    # sigma = pm.HalfNormal("sigma", sigma=1)
    # y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)


pm.model_to_graphviz(model)


# Inference / Model fitting
with model:
    # Inference
    idata = pm.sample(draws=1000, tune=1000, cores=4)

# Summarize the trace
az.plot_trace(idata)
az.summary(idata, hdi_prob=0.95)


# generate out-of-sample predictions
with model:
    pm.set_data(
        {
            "Task Performance (Observed)": X_test,
            "Task Performance (Unobserved)": y_test,
        }
    )
    idata.extend(pm.sample_posterior_predictive(idata))

p_test_pred = idata.posterior_predictive["y_obs"].mean(dim=["chain", "draw"])


def compare_with_linear_model(X_train, X_test, y_train, y_test, pymc_predictions):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    plt.scatter(y_test, y_pred, color="blue", label="Linear Regression")
    plt.scatter(y_test, pymc_predictions, color="red", label="PyMC")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    # plot the line of best fit
    comb = np.concatenate([y_test, y_pred, pymc_predictions])
    _min, _max = np.min(comb), np.max(comb)
    plt.plot([_min, _max], [_min, _max], color="red")
    plt.show()

    print("Pearson correlation between true and predicted values")
    print(f"LR: {pearsonr(y_test, y_pred)[0]}")
    print(f"PyMC: {pearsonr(y_test, pymc_predictions)[0]}")


compare_with_linear_model(
    X_train, X_test, y_train, y_test, pymc_predictions=p_test_pred.values
)
