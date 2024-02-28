"""Forward RCN layers."""
from typing import Optional, Union, Tuple, List, Callable, Literal
import torch
import torch.nn as nn
import numpy as np

from ..util import value_to_tuple, batched


class Linear(nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use
    :ref:`different precision<fp16_on_mi200>` for backward.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default = True
        If set to ``False``, the layer will not use an additive bias.

    Notes
    -----

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension are the
        same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes  # noqa
    ----------
    weight : torch.Tensor,
    shape = :math:`(\text{out\_features}, \text{in\_features})`.
        The hidden weights of the module.
    bias : torch.Tensor, shape = :math:`(\text{out\_features})`.
        The learnable bias of the module. If :attr:`bias` is ``True``, the
        values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{in\_features}}`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_scaling: float = 1., bias_scaling: float = 0.,
                 bias_shift: float = 0., input_sparsity: float = 0.9,
                 device: Optional[str] = None, dtype: Optional = None):
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.bias_shift = bias_shift
        self.input_sparsity = input_sparsity

        super().__init__(in_features=in_features, out_features=out_features,
                         bias=bias, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for name, weight in self.named_parameters():
            weight.requires_grad_(False)
            if "weight" in name:
                nn.init.sparse_(weight, sparsity=self.input_sparsity,
                                std=self.input_scaling)
            elif "bias" in name:
                nn.init.uniform_(weight, -self.bias_scaling + self.bias_shift,
                                 self.bias_scaling + self.bias_shift)

    def set_external_weights(self, weights: torch.Tensor) -> None:
        """
        Set externally initialized weights for the layer.

        Parameters
        ----------
        weights : torch.Tensor, shape = (out_features, in_features).
            The externally initialized weight tensor.
        """
        if weights.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Shape of weights ({weights.shape}) does not match the "
                f"expected shape ({(self.out_features, self.in_features)}).")

        with torch.no_grad():
            self.weight.copy_(weights)
            self.weight.requires_grad_(False)

    def set_external_bias(self, bias: torch.Tensor) -> None:
        """
        Set externally initialized bias for the layer.

        Parameters
        ----------
        bias : torch.Tensor, shape = (out_features, ).
            The externally initialized bias tensor.
        """
        if bias.shape != (self.out_features, ):
            raise ValueError(
                f"Shape of bias ({bias.shape}) does not match the expected "
                f"shape ({(self.out_features, )}).")

        with torch.no_grad():
            self.bias.copy_(bias)
            self.bias.requires_grad_(False)


class ELM(nn.Sequential):
    r"""
    Applies a multi-layer Extreme Learning Machine with :math:`\tanh` or an
    arbitrary non-linearity to an input.

    For each element in the input, each layer computes the following function:

    .. math::
        h = \tanh(x W_{ih}^T + b_{ih})

    where :math:`h` is the hidden state, and :math:`x` is the input.
    If :attr:`nonlinearity` is not :math:'\tanh', then the specific
    non-linearity is used instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_sizes : Tuple[int, ...]
        The number of features in each hidden layer `h`.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_scaling : Union[float, Tuple[float, ...]], default = 0.
        Scales the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_shift : Union[float, Tuple[float, ...]], default = 0.
        Shifts the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    input_sparsity : Union[float, Tuple[float, ...]], default = 0.9
        Ratio between zero and non-zero values in the input weights `w_ih`. If
        it is a Tuple, it needs to have `num_layer` entries for each layer. If
        it is a number, the value in each layer is the same.
    activation_layer : Optional[Callable[..., nn.Module]], default = nn.ReLU
        The non-linearity to use.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Leaky integration is not supported. However, leaky integrator deep RCN
    models can be built using the ``LeakyELM``.

    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}
    """

    def __init__(
            self, input_size: int, hidden_sizes: Tuple[int, ...],
            bias: bool = True,
            input_scaling: Union[float, Tuple[float, ...]] = 1.,
            bias_scaling: Union[float, Tuple[float, ...]] = 0.,
            bias_shift: Union[float, Tuple[float, ...]] = 0.,
            input_sparsity: Union[float, Tuple[float, ...]] = 0.9,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            device: Optional[str] = None, dtype: Optional = None):
        self.num_layers = len(hidden_sizes)
        self.input_scaling = value_to_tuple(input_scaling, self.num_layers)
        self.bias_scaling = value_to_tuple(bias_scaling, self.num_layers)
        self.bias_shift = value_to_tuple(bias_shift, self.num_layers)
        self.input_sparsity = value_to_tuple(input_sparsity, self.num_layers)
        layers = []
        in_features = input_size
        for layer_id, out_features in enumerate(hidden_sizes):
            layers.append(Linear(in_features, out_features, bias=bias,
                                 input_scaling=self.input_scaling[layer_id],
                                 bias_scaling=self.bias_scaling[layer_id],
                                 bias_shift=self.bias_shift[layer_id],
                                 input_sparsity=self.input_sparsity[layer_id],
                                 device=device, dtype=dtype))
            layers.append(activation_layer())
            in_features = out_features
        super().__init__(*layers)

    def set_external_weights(self, external_weights: List[torch.Tensor]) \
            -> None:
        """
        Set externally initialized weights for each layer.

        Parameters
        ----------
        external_weights : List[torch.Tensor].
            The externally initialized weight tensors.
        """
        k = 0
        for name, layer in self.named_modules():
            if name != "":
                if int(name) % 2 == 0:
                    layer.set_external_weights(external_weights[k])
                    k += 1

    def set_external_bias(self, external_bias: List[torch.Tensor]) -> None:
        """
        Set externally initialized bias for each layer.

        Parameters
        ----------
        external_bias : List[torch.Tensor].
            The externally initialized bias tensors.
        """
        k = 0
        for name, layer in self.named_modules():
            if name != "":
                if int(name) % 2 == 0:
                    layer.set_external_bias(external_bias[k])
                    k += 1


class LeakyELM(ELM):
    r"""
    Applies a multi-layer Extreme Learning Machine with :math:`\tanh` or an
    arbitrary non-linearity to an input.

    For each element in the input, each layer computes the following function:

    .. math::
        h = \tanh(x W_{ih}^T + b_{ih})

    where :math:`h` is the hidden state, and :math:`x` is the input.
    If :attr:`nonlinearity` is not :math:'\tanh', then the specific
    non-linearity is used instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_sizes : Tuple[int, ...]
        The number of features in each hidden layer `h`.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_scaling : Union[float, Tuple[float, ...]], default = 0.
        Scales the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_shift : Union[float, Tuple[float, ...]], default = 0.
        Shifts the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    input_sparsity : Union[float, Tuple[float, ...]], default = 0.9
        Ratio between zero and non-zero values in the input weights `w_ih`. If
        it is a Tuple, it needs to have `num_layer` entries for each layer. If
        it is a number, the value in each layer is the same.
    activation_layer : Optional[Callable[..., nn.Module]], default = nn.ReLU
        The non-linearity to use.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Leaky integration is not supported. However, leaky integrator deep RCN
    models can be built using the ``RCNCell``.

    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}
    """

    def __init__(
            self, input_size: int, hidden_sizes: Tuple[int, ...],
            bias: bool = True,
            input_scaling: Union[float, Tuple[float, ...]] = 1.,
            bias_scaling: Union[float, Tuple[float, ...]] = 0.,
            bias_shift: Union[float, Tuple[float, ...]] = 0.,
            input_sparsity: Union[float, Tuple[float, ...]] = 0.9,
            leakage:  Union[float, Tuple[float, ...]] = 1.0,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            device: Optional[str] = None, dtype: Optional = None):
        self.leakage = value_to_tuple(leakage, self.num_layers)
        super().__init__(
            input_size=input_size, hidden_sizes=hidden_sizes, bias=bias,
            input_scaling=input_scaling, bias_scaling=bias_scaling,
            bias_shift=bias_shift, input_sparsity=input_sparsity,
            activation_layer=activation_layer, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Forward function of the neural network. Pass the input features through
        the hidden layers and return the hidden layer states of the final
        layer.

        Parameters
        ----------
        input : Tensor, shape = :math:`(*, H_{in})`.
            Tensor containing the input features. The input must also be a
            packed variable length sequence. See
            :func:`torch.nn.utils.rnn.pack_padded_sequence` or
            :func:`torch.nn.utils.rnn.pack_sequence` for details.

        Returns
        -------
        output : Tensor, shape = :math:`(*, H_{out})`.
            Tensor containing the output features `(h_t)` from the last layer
            of the ELM.
        """
        if input.dim() not in (1, 2):
            raise ValueError(f"LeakyELM: Expected input to be 1D or 2D, "
                             f"got {input.shape}D instead")
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        batch_size = input.shape[0]
        layer_id = 0
        layer_output = torch.zeros_like(input)
        for weights, activation in batched(self.modules(), n=2):
            out_features = weights.out_features
            layer_output = torch.zeros((batch_size + 1, out_features))
            for k in range(batch_size):
                layer_output[k+1] = \
                    (1 - self.leakage[layer_id]) * layer_output[k] + \
                    self.leakage[layer_id] * activation(weights(input[k]))
            input = layer_output[1:]
            layer_id += 1
        return layer_output[1:]
