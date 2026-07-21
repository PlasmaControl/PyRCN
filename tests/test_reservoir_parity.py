"""Bit-exact parity gate for the optimized reservoir forward.

A fixture (``tests/fixtures/reservoir_parity.pt``) captures the reservoir
outputs of the *original* per-timestep Python loop over a grid of configs.
This test regenerates the outputs with the current (optimized) code and asserts
they match the fixture:

* all configs must match the fixture to ``atol=rtol=1e-12`` (the agreed
  tolerance). The scripted/general path is bit-exact to the original loop on a
  fixed library build, but the fixture is captured on one numpy/torch build and
  compared on possibly-different ones (e.g. across CI Python versions), so a
  ~1-ULP difference is expected and tolerated;
* the fused ATen sub-case (``leakage == 1``, ``tanh``/``relu``, plain
  reservoir) uses a different kernel, likewise matched to ``atol=rtol=1e-12``.

Run this module as a script on the *unmodified* code to (re)create the
fixture::

    .venv/bin/python tests/test_reservoir_parity.py
"""
from __future__ import annotations

import os

import torch

from pyrcn.nn import EulerReservoir, Reservoir

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "reservoir_parity.pt")

ACTS = ["tanh", "relu", "logistic", "identity", "bounded_relu"]
SHAPES = [(1, 30, 12), (3, 20, 8)]
SEEDS = [0, 1]
DTYPE = torch.float64


def _weights(hidden, seed):
    g = torch.Generator().manual_seed(1000 + seed * 7 + hidden)
    return torch.randn((hidden, hidden), generator=g, dtype=DTYPE) * 0.1


def _inputs(shape, seed):
    g = torch.Generator().manual_seed(2000 + seed)
    return torch.randn(shape, generator=g, dtype=DTYPE) * 0.3


def _initial(batch, hidden, seed):
    g = torch.Generator().manual_seed(3000 + seed)
    return torch.randn((batch, hidden), generator=g, dtype=DTYPE) * 0.2


def is_fused(cfg):
    """Configs that take the fused ATen sub-case."""
    return (cfg["model"] == "reservoir" and cfg["leakage"] == 1.0
            and cfg["activation"] in ("tanh", "relu"))


def _configs():
    for act in ACTS:
        for leakage in (1.0, 0.6):
            for bidir in (False, True):
                inits = (None,) if bidir else (None, "nonzero")
                for shape, seed in [(s, sd) for s in SHAPES for sd in SEEDS]:
                    for init in inits:
                        yield {
                            "model": "reservoir", "activation": act,
                            "leakage": leakage, "bidirectional": bidir,
                            "shape": shape, "seed": seed, "init": init,
                        }
    for act in ACTS:
        for epsilon in (0.01, 0.05):
            for shape, seed in [(s, sd) for s in SHAPES for sd in SEEDS]:
                for init in (None, "nonzero"):
                    yield {
                        "model": "euler", "activation": act,
                        "epsilon": epsilon, "shape": shape, "seed": seed,
                        "init": init,
                    }


def _key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def _build_and_run(cfg):
    batch, length, hidden = cfg["shape"]
    weights = _weights(hidden, cfg["seed"])
    x = _inputs(cfg["shape"], cfg["seed"])
    if cfg["init"] == "nonzero":
        initial_state = _initial(batch, hidden, cfg["seed"])
    else:
        initial_state = None
    if cfg["model"] == "reservoir":
        res = Reservoir(
            hidden_size=hidden, spectral_radius=0.9, leakage=cfg["leakage"],
            activation=cfg["activation"],
            bidirectional=cfg["bidirectional"], dtype=DTYPE)
    else:
        res = EulerReservoir(
            hidden_size=hidden, recurrent_scaling=0.9, gamma=0.01,
            epsilon=cfg["epsilon"], activation=cfg["activation"], dtype=DTYPE)
    res.set_recurrent_weights(weights)
    with torch.no_grad():
        states, final = res(x, initial_state)
    return states, final


def generate_outputs():
    out = {}
    for cfg in _configs():
        states, final = _build_and_run(cfg)
        out[_key(cfg)] = {
            "cfg": cfg, "states": states, "final": final,
            "fused": is_fused(cfg),
        }
    return out


def _load_fixture():
    return torch.load(FIXTURE, weights_only=False)


def test_reservoir_parity():
    fixture = _load_fixture()
    fresh = generate_outputs()
    assert set(fresh) == set(fixture), "grid changed vs fixture"
    max_exact = 0.0
    max_fused = 0.0
    for key, ref in fixture.items():
        got = fresh[key]
        for field in ("states", "final"):
            a, b = got[field], ref[field]
            assert a.shape == b.shape, f"{key}:{field} shape"
            if ref["fused"]:
                diff = (a - b).abs().max().item() if a.numel() else 0.0
                max_fused = max(max_fused, diff)
                assert torch.allclose(a, b, atol=1e-12, rtol=1e-12), (
                    f"fused config {key}:{field} diff={diff}")
            else:
                diff = (a - b).abs().max().item() if a.numel() else 0.0
                max_exact = max(max_exact, diff)
                assert torch.allclose(a, b, atol=1e-12, rtol=1e-12), (
                    f"general config {key}:{field} not within 1e-12 "
                    f"(diff={diff})")
    print(f"max general-path diff (<=1e-12): {max_exact}")
    print(f"max fused-path diff (<=1e-12): {max_fused}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    torch.save(generate_outputs(), FIXTURE)
    n = len(_load_fixture())
    print(f"wrote {n} configs to {FIXTURE}")
