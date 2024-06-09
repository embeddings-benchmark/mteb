"""For this section we seek to create a (PyMC3) model to predict the performance on an unseen task $t_i$.

We denote the performance on the unseen task $S_{t_i}$ and the performance on the seen tasks $S_{t_j}$, where $t_j \in T_{-t_i}$.

The model takes the general form:

$S_{t_i} \sim Beta(\alpha_{t_i}, \beta_{t_i})$

Where $\alpha_{t_i}$ and $\beta_{t_i}$ are the parameters of the Beta distribution, defined as:

$\alpha_{t_i} = y_{-t_i} * \phi$
$\beta_{t_i} = (1 - y_{-t_i}) * \phi$

Where $\phi$ is the concentration parameter, and $y_{-t_i}$ takes the general form:

$y_{-t_i} = f(S_{t_{j \neq t_i}}; \theta)$

For which we assume a linear combination of the performances on the seen tasks, where $\theta_{t_i}$ is the weight of the seen tasks in the prediction of the unseen task:

$y_{-t_i} = \sum_{t_j \in T_{-t_i}} \beta_{t_j} * S_{t_j}$

Where $\beta_{t_j} \in \theta$.


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
# pytensor.config.optimizer = "None"  # type: ignore
pytensor.config.optimizer = "fast_compile"  # type: ignore

# columns = tasks, rows = embedding methods
data = pd.DataFrame(
    {
        "task_A": [0.8, 0.7, 0.6, 0.5],
        "task_B": [0.6, 0.6, 0.6, 0.6],
        "task_C": [0.9, 0.8, 0.7, 0.6],
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
    X = pm.MutableData("t_o", x_train)  # task performance on seen tasks
    y = pm.MutableData(
        "t_u", y_train
    )  # task performance on unseen task (to be predicted)

    # Priors
    beta = pm.Normal("beta", mu=0, sigma=1, dims="coeffs")

    # Task-specific parameters (example: difficulty, variability)
    task_difficulty = pm.Normal("task_difficulty", mu=0, sigma=1)
    task_variability = pm.Normal("task_variability", mu=0, sigma=1)

    # linear model
    mu = pm.math.dot(X, beta)  # type: ignore

    alpha = pm.math.exp(mu + task_difficulty)  # type: ignore
    beta = pm.math.exp((1 - mu) + task_variability)  # type: ignore

    # Likelihood
    y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta, observed=y_train)

pm.model_to_graphviz(model)

# Inference / Model fitting
with model:
    # Inference
    idata = pm.sample(1000, tune=1000, cores=4)

# Summarize the trace
az.plot_trace(idata)
az.summary(idata, hdi_prob=0.95)


# # generate out-of-sample predictions
with model:
    pm.set_data({"t_o": x_test, "t_u": y_test})
    idata.extend(pm.sample_posterior_predictive(idata))


# generate out-of-sample predictions with manual calculation
with model:
    # Priors
    beta = idata.posterior["beta"].mean(axis=0)
    task_difficulty = idata.posterior["task_difficulty"].mean()
    task_variability = idata.posterior["task_variability"].mean()

    # linear model
    mu = np.dot(beta, x_test)
    alpha = np.exp(mu + task_difficulty.values)
    beta = np.exp((1 - mu) + task_variability.values)

    # Likelihood
    y_obs = np.random.beta(alpha, beta)

    print(y_obs)
