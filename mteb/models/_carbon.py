"""Estimate inference carbon cost (gCO₂ per million tokens) from active parameters.

The number of active parameters is enough to give a rough, order-of-magnitude
estimate of inference cost. We follow the EcoLogits linear-regression structure
(energy per token is roughly linear in the number of active parameters) so that
MTEB estimates sit on the same scale as EcoLogits figures for closed generative
models::

    energy_per_token (Wh) = ENERGY_ALPHA * active_params_billions + ENERGY_BETA

Every figure reflects *benchmark* conditions, not real-world deployment: we have
no visibility into deployment hardware or grid carbon intensity, so estimates
should always be surfaced with a leading "~" and accompanied by the assumptions
in `CO2_ASSUMPTIONS`.

References:
    - EcoLogits methodology: https://ecologits.ai/0.2/methodology/llm_inference/
    - Luccioni et al. (2024), "Power Hungry Processing: Watts Driving the Cost of
      AI Deployment?", https://arxiv.org/abs/2311.16863
    - Luccioni et al. (2022), "Estimating the Carbon Footprint of BLOOM",
      https://jmlr.org/papers/volume24/23-0069/23-0069.pdf
"""

from __future__ import annotations

# EcoLogits LLM-inference energy model (GPU), fitted on A100 80GB.
ENERGY_ALPHA_WH_PER_TOKEN_PER_B = 8.91e-5  # Wh / token / billion active params
ENERGY_BETA_WH_PER_TOKEN = 1.43e-3  # Wh / token (fixed overhead)

# Grid / datacenter assumptions surfaced alongside every estimate.
BENCHMARK_HARDWARE = "A100 80GB"
CARBON_INTENSITY_G_PER_KWH = 400.0  # gCO₂/kWh, ADEME world average
PUE = 1.2  # power usage effectiveness

CO2_ASSUMPTIONS: dict[str, str] = {
    "benchmark_hardware": BENCHMARK_HARDWARE,
    "carbon_intensity": f"{CARBON_INTENSITY_G_PER_KWH:.0f} gCO₂/kWh (ADEME world)",
    "pue": f"{PUE}",
    "note": (
        "Estimate reflects benchmark conditions, not real-world deployment. "
        "Derived from active parameters via the EcoLogits linear model."
    ),
}


def estimate_co2_per_million_tokens(n_active_parameters: int | None) -> float | None:
    """Estimate inference carbon cost in gCO₂ per million tokens.

    Args:
        n_active_parameters: Number of active parameters used at inference time.
            For dense models this is ``n_parameters - n_embedding_parameters``;
            for MoE models it is the number of parameters actually activated.

    Returns:
        The estimated grams of CO₂ per million tokens, or None when the active
        parameter count is unknown.
    """
    if n_active_parameters is None:
        return None
    active_params_b = n_active_parameters / 1e9
    energy_wh_per_token = (
        ENERGY_ALPHA_WH_PER_TOKEN_PER_B * active_params_b + ENERGY_BETA_WH_PER_TOKEN
    )
    # Wh/token -> kWh per million tokens, adjusted for datacenter overhead (PUE).
    energy_kwh_per_million = energy_wh_per_token * 1e6 / 1000 * PUE
    return energy_kwh_per_million * CARBON_INTENSITY_G_PER_KWH
