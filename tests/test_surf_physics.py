"""Wave-physics building blocks: dispersion, breaking, sheltering, wind chop, currents."""
import numpy as np
import pandas as pd
import pytest

from app.features import surf as phys

G = phys.G


def one(H, T, d, sigma=0.5):
    return [{"H": np.array([H]), "T": np.array([T]), "dir": np.array([d]), "sigma": sigma}]


# ----------------------------------------------------------------- linear theory
def test_dispersion_relation_residual_and_limits():
    T = np.array([4.0, 8.0, 12.0, 16.0])
    for h in (2.0, 10.0, 40.0):
        k = phys.wavenumber(T, h)
        w = 2 * np.pi / T
        assert np.allclose(G * k * np.tanh(k * h), w ** 2, rtol=1e-6)
    C, Cg = phys.celerity(T, np.inf)                      # deep water
    assert np.allclose(C, G * T / (2 * np.pi)) and np.allclose(Cg, C / 2)
    C, _ = phys.celerity(np.array([30.0]), 1.0)           # shallow water
    assert C[0] == pytest.approx(np.sqrt(G * 1.0), rel=0.01)


def test_feet_conversion_constant():
    assert phys.FT_PER_M == pytest.approx(3.28084)


# ------------------------------------------------------------------ breaking
def test_head_on_breaking_matches_komar_gaughan():
    H0, T = 1.5, 9.0
    hb = phys.breaking_from_partitions(one(H0, T, 180.0), 180.0, None, None)["hb_m"][0]
    expected = 0.39 * G ** 0.2 * (T * H0 ** 2) ** 0.4
    assert hb == pytest.approx(expected, rel=0.02)


def test_breaking_is_not_unrealistically_low_for_head_on_swell():
    # A 5 ft, 9-10 s swell aimed at the beach must give surf at least as big as offshore Hs.
    for T in (8.0, 10.0, 12.0):
        hb = phys.breaking_from_partitions(one(1.5, T, 175.0, sigma=15), 180.0, None, 22.0)["hb_m"][0]
        assert hb >= 1.0 * 1.5, T


def test_obliquity_reduces_and_offshore_waves_give_nothing():
    head = phys.breaking_from_partitions(one(1.5, 9, 180.0), 180.0, None, None)["hb_m"][0]
    obl = phys.breaking_from_partitions(one(1.5, 9, 120.0), 180.0, None, None)["hb_m"][0]
    away = phys.breaking_from_partitions(one(1.5, 9, 0.0), 180.0, None, None)["hb_m"][0]
    assert 0 < obl < head
    assert away == pytest.approx(0.0, abs=1e-6)


def test_partition_energies_combine_not_heights():
    assert phys.combine_hs(np.array([3.0]), np.array([4.0]))[0] == pytest.approx(5.0)
    assert np.isnan(phys.combine_hs(np.array([np.nan]), np.array([np.nan]))[0])
    two = phys.breaking_from_partitions(one(1.0, 9, 180) + one(1.0, 9, 180), 180, None, None)["hb_m"][0]
    single = phys.breaking_from_partitions(one(np.sqrt(2.0), 9, 180), 180, None, None)["hb_m"][0]
    assert two == pytest.approx(single, rel=1e-6)          # energy, not height, adds


def test_short_wind_waves_do_not_explode():
    # regression: 1.7 s chop at grazing angle used to give ~8 ft "surf" from 1 ft waves
    parts = one(0.6, 9.0, 160.0, 15) + one(0.3, 1.7, 255.0, 30)
    hb = phys.breaking_from_partitions(parts, 185.0, None, 24.0)["hb_m"][0]
    alone = phys.breaking_from_partitions(one(0.6, 9.0, 160.0, 15), 185.0, None, 24.0)["hb_m"][0]
    assert hb < alone * 1.2


def test_breaker_index_limits():
    assert phys.breaker_index(np.nan, 1.0, 8.0) == pytest.approx(0.78)
    steep = phys.breaker_index(0.05, 1.0, 8.0)
    assert 0.78 < steep < 1.2


# ---------------------------------------------------------------- sheltering
def _geom(blocked_from: float, blocked_to: float, fetch=1.0):
    dirs = np.arange(0, 360, 10.0)
    inside = (dirs >= blocked_from) & (dirs <= blocked_to)
    land_side = (dirs < 90) | (dirs > 270)             # spot faces south: land to the north
    f = np.where(land_side, 0.0, np.where(inside, fetch, 100.0))
    return phys.Geometry(dirs, f, np.full(36, 5.0), f >= 100, np.full(36, 5.0))


def test_shadow_with_diffraction_edge():
    g = _geom(90, 140)
    k = phys.wavenumber(np.array([9.0]), 5.0)
    open_w = g.diffraction_weight(np.array([180.0]), k)[0]
    edge = g.diffraction_weight(np.array([140.0]), k)[0]
    deep = g.diffraction_weight(np.array([100.0]), k)[0]
    assert open_w == 1.0
    assert 0 < deep < edge <= 0.5
    # waves refract toward the shore before reaching the land, so test with a wide
    # shadow like Point Lookout's (Jones Beach West End blocks E to SSE)
    g2 = _geom(90, 160)
    sheltered = phys.breaking_from_partitions(one(1.5, 9, 120.0, 15), 180.0, g2, 22.0)["hb_m"][0]
    exposed = phys.breaking_from_partitions(one(1.5, 9, 120.0, 15), 180.0, None, 22.0)["hb_m"][0]
    assert sheltered < 0.9 * exposed


def test_effective_fetch_averages_neighbouring_rays():
    g = _geom(0, -1)                                   # nothing extra blocked
    f, d = g.effective_fetch(np.array([180.0, 0.0]))
    assert f[0] == pytest.approx(100.0) and f[1] == 0.0
    assert d[1] == 0.0


# ---------------------------------------------------------------- wind chop
def test_fetch_limited_deep_water_limit_and_zero_cases():
    U, F = 10.0, 20_000.0
    H, T = phys.fetch_limited(np.array([U]), np.array([F]), np.array([200.0]))
    UA = 0.71 * U ** 1.23
    assert H[0] == pytest.approx(1.6e-3 * np.sqrt(G * F / UA ** 2) * UA ** 2 / G, rel=0.05)
    assert 2 < T[0] < 6
    H0, _ = phys.fetch_limited(np.array([U]), np.array([0.0]), np.array([5.0]))
    Hn, _ = phys.fetch_limited(np.array([np.nan]), np.array([F]), np.array([5.0]))
    Hs, _ = phys.fetch_limited(np.array([U]), np.array([F]), np.array([1.0]))
    assert H0[0] == 0.0 and np.isnan(Hn[0]) and Hs[0] < H[0]


def test_long_island_sound_chop_is_realistic():
    # NE 15 kt over ~30 km of the Sound: NWS typically forecasts 2-3 ft.
    H, _ = phys.fetch_limited(np.array([15 * 0.5144]), np.array([30_000.0]), np.array([20.0]))
    assert 1.5 <= H[0] * phys.FT_PER_M <= 3.5


# ----------------------------------------------------------- currents, power
def test_opposing_current_steepens_and_blocks():
    f0, b0 = phys.opposing_current_factor(np.array([8.0]), np.array([0.0]))
    f1, b1 = phys.opposing_current_factor(np.array([8.0]), np.array([1.0]))
    f2, b2 = phys.opposing_current_factor(np.array([4.0]), np.array([1.6]))
    assert f0[0] == pytest.approx(1.0) and not b0[0]
    assert f1[0] > 1.05 and not b1[0]
    assert b2[0]


def test_wave_power_and_breaker_type():
    assert phys.wave_power_kw_m(1.0, 8.0) == pytest.approx(1025 * G ** 2 * 8 / (64 * np.pi) / 1000)
    assert phys.breaker_type(np.array([0.2, 1.0, 3.0])).tolist() == ["spilling", "plunging", "surging"]


def test_trend_labels():
    ix = pd.date_range("2026-10-23", periods=24, freq="1h", tz="UTC")
    up = phys.trend(pd.Series(np.linspace(1, 5, 24), index=ix))
    flat = phys.trend(pd.Series(np.full(24, 2.0), index=ix))
    down = phys.trend(pd.Series(np.linspace(5, 1, 24), index=ix))
    assert up.iloc[0] == "building" and flat.iloc[0] == "steady" and down.iloc[0] == "subsiding"
