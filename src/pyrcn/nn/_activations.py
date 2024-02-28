"""The :mod:`activations` contains various activation functions for PyRCN."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from torch import nn


ACTIVATIONS = {
    "elu": nn.ELU,
    "hard_shrink": nn.Hardshrink,
    "hard_sigmoid": nn.Hardsigmoid,
    "hard_tanh": nn.Hardtanh,
    "hard_swish": nn.Hardswish,
    "identity": nn.Identity,
    "leaky_relu": nn.LeakyReLU,
    "log_sigmoid": nn.LogSigmoid,
    "relu": nn.ReLU,
    "relu_6": nn.ReLU6,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "tanh_shrink": nn.Tanhshrink,
}
