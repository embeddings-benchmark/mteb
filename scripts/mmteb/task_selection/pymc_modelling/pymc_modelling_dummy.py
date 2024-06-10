"""For more on the description of this see the corresponding markdown


Requires:
    pip install pymc
    pip install graphviz
    (plus the graphviz library installed on your system, e.g. installed using `brew install graphviz`)
"""

# first we create the simplle example where we predict the performance of task C, given the performances on tasks A and B
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor


# Set PyTensor configurations for better debugging
pytensor.config.exception_verbosity = "high"  # type: ignore
pytensor.config.optimizer = "None"  # type: ignore

# columns = tasks, rows = embedding methods
data = pd.DataFrame(
    {
        "task_A": [0.8, 0.7, 0.6, 0.5, 0.7, 0.6, 0.5],
        "task_B": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        "task_C": [0.9, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6],
    }
)

# Prepare the data for modelling
labels = ["task_A", "task_B"]
# labels.append("Intercept")
# data["Intercept"] = 1
x_train = data[labels].to_numpy()
y_train = data["task_C"].to_numpy()

x_test = np.array([[0.4, 0.6], [0.8, 0.6]])
y_test = np.array([0.5, 0.9])


# Define the model
coords = {"coeffs": labels}
with pm.Model(coords=coords) as model:
    X = pm.MutableData("Task Performance (Observed)", x_train) 
    y = pm.MutableData(
        "Task Performance (Unobserved)", y_train
    )  # task performance on unseen task (to be predicted)

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")

    # Linear combination of inputs
    mu_linear = pm.math.dot(X, beta)  # type: ignore

    # Logit link function to ensure mu is between 0 and 1
    mu = pm.math.invlogit(mu_linear)
    # Precision parameter (optional to estimate or set as a constant)
    phi = pm.Gamma("phi", alpha=2, beta=2)

    # Likelihood
    y_obs = pm.Beta("y_obs", alpha=mu * phi, beta=(1 - mu) * phi, observed=y)

pm.model_to_graphviz(model)


# Inference / Model fitting
with model:
    # Inference
    idata = pm.sample(1000, tune=1000)

# Summarize the trace
az.plot_trace(idata)
az.summary(idata, hdi_prob=0.95)


# generate out-of-sample predictions
with model:
    pm.set_data({"Task Performance (Observed)": x_test, "Task Performance (Unobserved)": y_test})
    idata.extend(pm.sample_posterior_predictive(idata))

p_test_pred = idata.posterior_predictive["y_obs"].mean(dim=["chain", "draw"])
