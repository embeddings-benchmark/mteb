For this section we seek to create a (PyMC3) model to predict the performance on an unseen task $t_i$.

We denote the performance on the unseen task $S_{t_i}$ and the performance on the seen tasks $S_{t_j}$, where $t_j \in T_{-t_i}$.

The model takes the general form:

$S_{t_i} \sim Beta(\alpha_{t_i}, \beta_{t_i})$

Where $\alpha_{t_i}$ and $\beta_{t_i}$ are the parameters of the Beta distribution, defined as:

$\alpha_{t_i} = y_{-t_i} * \phi$
$\beta_{t_i} = (1 - y_{-t_i}) * \phi$

Where $\phi$ is the concentration parameter, and $y_{-t_i}$ takes the general form:

$y_{-t_i} = f(S_{t_{j \neq t_i}}; \theta)$

For which we assume a linear combination of the performances on the seen tasks, where $\theta_{t_i}$ is the weight of the seen tasks in the prediction of the unseen task:

$y_{-t_i} = \sum_{t_j \in T_{-t_i}} \beta_{t_j} \cdot S_{t_j}$

Where $\beta_{t_j} \in \theta$.


