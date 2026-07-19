"""The torch backend modules run on CPU and honour the ``device`` argument.

The estimators are CPU/float64 today; device selection is a backend-module
capability (``InputFeatureMap`` / ``Reservoir`` / ``IncrementalRidge`` all take
``device=``). The CUDA test is skipped when no GPU is available.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyrcn.backend import InputFeatureMap, IncrementalRidge, Reservoir


def _run_pipeline(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature map -> reservoir -> readout on ``device``; return outputs."""
    n_features, hidden, length = 3, 8, 12
    fm = InputFeatureMap(n_features, hidden, dtype=torch.float64,
                         device=device)
    x = torch.randn(length, n_features, dtype=torch.float64, device=device)
    feats = fm(x)

    res = Reservoir(hidden, spectral_radius=0.9, leakage=0.6,
                    dtype=torch.float64, device=device)
    res.set_recurrent_weights(np.eye(hidden))
    states, final = res(feats.unsqueeze(0))
    states = states.squeeze(0)

    readout = IncrementalRidge(dtype=torch.float64, device=device)
    y = torch.randn(length, 2, dtype=torch.float64, device=device)
    readout.fit(states, y)
    pred = readout.predict(states)
    return states, pred


def test_backend_runs_on_cpu() -> None:
    states, pred = _run_pipeline("cpu")
    assert states.device.type == "cpu"
    assert pred.device.type == "cpu"
    assert states.shape == (12, 8)
    assert pred.shape == (12, 2)
    assert torch.isfinite(pred).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_backend_runs_on_cuda() -> None:
    states, pred = _run_pipeline("cuda")
    assert states.device.type == "cuda"
    assert pred.device.type == "cuda"
    assert torch.isfinite(pred).all()
