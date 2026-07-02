import mteb
from mteb.models._carbon import (
    CARBON_INTENSITY_G_PER_KWH,
    ENERGY_ALPHA_WH_PER_TOKEN_PER_B,
    ENERGY_BETA_WH_PER_TOKEN,
    PUE,
    estimate_co2_per_million_tokens,
)
from mteb.models.model_meta import ModelMeta


def test_estimate_co2_none_when_active_params_unknown():
    assert estimate_co2_per_million_tokens(None) is None


def test_estimate_co2_matches_ecologits_linear_model():
    n_active = 100_000_000  # 0.1B active parameters
    active_b = n_active / 1e9
    energy_wh_per_token = (
        ENERGY_ALPHA_WH_PER_TOKEN_PER_B * active_b + ENERGY_BETA_WH_PER_TOKEN
    )
    expected = energy_wh_per_token * 1e6 / 1000 * PUE * CARBON_INTENSITY_G_PER_KWH

    assert estimate_co2_per_million_tokens(n_active) == expected


def test_estimate_co2_monotonic_in_active_params():
    small = estimate_co2_per_million_tokens(10_000_000)
    large = estimate_co2_per_million_tokens(7_000_000_000)
    assert small is not None and large is not None
    assert large > small


def test_model_meta_co2_property_uses_active_parameters():
    meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
    assert meta.n_active_parameters is not None
    assert meta.co2_cost_per_million_tokens == estimate_co2_per_million_tokens(
        meta.n_active_parameters
    )


def test_model_meta_co2_property_none_without_active_parameters():
    meta = ModelMeta.create_empty()
    assert meta.n_active_parameters is None
    assert meta.co2_cost_per_million_tokens is None
