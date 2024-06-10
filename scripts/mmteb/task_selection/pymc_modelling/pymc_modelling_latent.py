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
labels = [col for col in data.columns]
labels = labels[:10]
X_train, X_test = train_test_split(
    data[labels], test_size=0.20, random_state=42
)

# convert to 0-1 values
X_train = X_train / 100
X_test = X_test / 100


# Define the model assuming a latent variable g
# where g -> task_performance but it unobserved
coords = {"coeffs": labels}
with pm.Model(coords=coords) as model:
    g = pm.Normal("g", mu=0, sigma=1)
    phi = pm.Gamma("phi", alpha=2, beta=2)
    

    for task in labels:
        # create observed variable
        task_score = pm.MutableData(task, X_train[task])

        task_intercept = pm.Normal(f"{task}_intercept", mu=0, sigma=1)

        mu_linear = g + task_intercept
        mu = pm.math.invlogit(mu_linear)

        alpha = mu * phi
        beta = (1 - mu) * phi
        
        y = pm.Beta(f"{task}_y_obs", alpha=alpha, beta=beta, observed=task_score)    

pm.model_to_graphviz(model)
model.debug()


# Inference / Model fitting
with model:
    # Inference
    idata = pm.sample(draws=1000, tune=1000, cores=4)


# Summarize the trace
az.plot_trace(idata)
az.summary(idata, hdi_prob=0.95)


# generate predictions
observed_labels = [col for col in labels if col != task_to_predict]
with model:

    for task in observed_labels:
        model[task].observations = X_test[task]


assert X_test[[task_to_predict]].shape == p_test_pred.shape

def compare_with_linear_model(_X_train, _X_test, _y_train, _y_test, pymc_predictions):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(_X_train, _y_train)
    y_pred = model.predict(_X_test)
    y_pred = y_pred.flatten()
    _y_test = _y_test.values.flatten()

    plt.scatter(_y_test, y_pred, color="blue", label="Linear Regression")
    plt.scatter(_y_test, pymc_predictions, color="red", label="PyMC")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    # plot the line of best fit
    comb = np.concatenate([_y_test, y_pred, pymc_predictions])
    _min, _max = np.min(comb), np.max(comb)
    plt.plot([_min, _max], [_min, _max], color="red")
    plt.show()

    print("Pearson correlation between true and predicted values")
    print(f"LR: {pearsonr(_y_test, y_pred)[0]}")
    print(f"PyMC: {pearsonr(_y_test, pymc_predictions)[0]}")


compare_with_linear_model(
    _X_train = X_train[observed_labels], 
    _X_test = X_test[observed_labels], 
    _y_train = X_train[[task_to_predict]], 
    _y_test = X_test[[task_to_predict]], 
    pymc_predictions=p_test_pred.values
)


