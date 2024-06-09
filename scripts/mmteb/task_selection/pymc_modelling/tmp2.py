import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


# Creating synthetic data
np.random.seed(42)
n_samples = 100
X = np.random.normal(size=n_samples)
noise = np.random.normal(scale=0.1, size=n_samples)
y = (0.5 + 0.3 * X + noise)

# Transform y to be in (0, 1)
y = (y - y.min()) / (y.max() - y.min())
y = np.clip(y, 0.01, 0.99)  # avoid values exactly 0 or 1

with pm.Model() as model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    
    # Expected value of outcome, use a logit link function
    mu = pm.math.invlogit(alpha + beta * X)
    
    # Precision parameter (inverse of variance)
    kappa = pm.Gamma("kappa", alpha=0.1, beta=0.1)
    
    # Likelihood (sampling distribution) of observations
    y_obs = pm.Beta("y_obs", alpha=mu * kappa, beta=(1 - mu) * kappa, observed=y)

    # Posterior distribution
    trace = pm.sample(1000)


# Summarize the trace
az.plot_trace(trace)
az.summary(trace, hdi_prob=0.95)
