"""Recurrent RCN layers."""
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from . import init
from ..util import value_to_tuple


class ESN(nn.RNN):
    r"""
    Applies a multi-layer Echo State Network with :math:`\tanh` or
    :math:`\text{ReLU}` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T)

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used
    instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int, default = 1
        Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
        stacking two RCNs together to form a `deep RCN`, with the second RCN
        taking in outputs of the first RCN and computing the final results.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    spectral_radius : Union[float, Tuple[float, ...]], default = 1.
        Scales the recurrent weights `w_hh`. If it is a Tuple, it needs to have
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
    recurrent_sparsity : Union[float, Tuple[float, ...]], default = 0.9
        Ratio between zero and non-zero values in the recurrent weights `w_hh`.
        If it is a Tuple, it needs to have `num_layer` entries for each layer.
        If it is a number, the value in each layer is the same.
    batch_first : bool, default = False
        If ``True``, then the input and output tensors are provided as
        `(batch, seq, feature)` instead of `(seq, batch, feature)`. Note that
        this does not apply to hidden or cell states. See the Inputs/Outputs
        sections below for details.
    bidirectional:  bool, default = False
        If ``True``, becomes a bidirectional RCN.
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
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih_l[k] : torch.Tensor, shape = `(hidden_size, input_size)` or
    `(hidden_size, num_directions * hidden_size)`.
        The input-hidden weights of the k-th layer of shape
        `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
        `(hidden_size, num_directions * hidden_size)`.
    weight_hh_l[k] : torch.Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of the k-th layer of shape
        `(hidden_size, hidden_size)`.
    bias_ih_l[k] : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of the k-th layer of shape
        `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True,
                 input_scaling: Union[float, Tuple[float, ...]] = 1.,
                 spectral_radius: Union[float, Tuple[float, ...]] = 1.,
                 bias_scaling: Union[float, Tuple[float, ...]] = 0.,
                 bias_shift: Union[float, Tuple[float, ...]] = 0.,
                 input_sparsity: Union[float, Tuple[float, ...]] = 0.9,
                 recurrent_sparsity: Union[float, Tuple[float, ...]] = 0.9,
                 batch_first: bool = False, bidirectional: bool = False,
                 device: Optional[str] = None, dtype: Optional = None):
        self.input_scaling = value_to_tuple(input_scaling, num_layers)
        self.spectral_radius = value_to_tuple(spectral_radius, num_layers)
        self.bias_scaling = value_to_tuple(bias_scaling, num_layers)
        self.bias_shift = value_to_tuple(bias_shift, num_layers)
        self.input_sparsity = value_to_tuple(input_sparsity, num_layers)
        self.recurrent_sparsity = value_to_tuple(recurrent_sparsity,
                                                 num_layers)

        super().__init__(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, nonlinearity=nonlinearity, bias=bias,
            batch_first=batch_first, dropout=0.0, bidirectional=bidirectional,
            device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for name, weight in self.named_parameters():
            _, _, layer = name.split("_")
            layer_idx = int(layer[1:])
            weight.requires_grad_(False)
            if "weight_ih" in name:
                nn.init.sparse_(
                    weight, sparsity=self.input_sparsity[layer_idx],
                    std=self.input_scaling[layer_idx])
            elif "weight_hh" in name:
                nn.init.sparse_(
                    weight, sparsity=self.recurrent_sparsity[layer_idx],
                    std=1.)
                init.spectral_norm_(weight)
                weight *= self.spectral_radius[layer_idx]
            elif "bias_ih" in name:
                nn.init.uniform_(
                    weight, -(self.bias_scaling + self.bias_shift)[layer_idx],
                    (self.bias_scaling + self.bias_shift)[layer_idx])
            elif "bias_hh" in name:
                nn.init.zeros_(weight)
            else:
                print(name)

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward function of the neural network. Pass the input features through
        the hidden layers and return the hidden layer states of the final
        layer.

        Parameters
        ----------
        input : Tensor, shape = :math:`(L, H_{in})` or :math:`(L, N, H_{in})`
                        or :math:`(N, L, H_{in})`.
            Tensor containing the input features. The shape for unbatched
            input is :math:`(L, H_{in})`. For batched input, the shape is
            :math:`(L, N, H_{in})` when ``batch_first=False``, or
            :math:`(N, L, H_{in})` when ``batch_first=True``. The input can
            also be a packed variable length sequence. See
            :func:`torch.nn.utils.rnn.pack_padded_sequence` or
            :func:`torch.nn.utils.rnn.pack_sequence` for details.
        hx : Optional[Tensor], shape :math:`(D * \text{num\_layers}, H_{out})`
                               or :math:`(D * \text{num\_layers}, N, H_{out})
            Tensor containing the initial hidden state. The shape for unbatched
            input is :math:`(D * \text{num\_layers}, H_{out})`. For batched
            input, the shape is :math:`(D * \text{num\_layers}, N, H_{out})`.
            Defaults to zero if not provided.

        Returns
        -------
        output : Tensor, shape = :math:`(L, D * H_{out})` or
        :math:`(L, N, D * H_{out})` or :math:`(N, L, D * H_{out})`.
            Tensor containing the output features `(h_t)` from the last layer
            of the RNN, for each `t`. The shape for unbatched input is
            :math:`(L, D * H_{out})`. For batched input, the shape is
            :math:`(L, N, D * H_{out})` when ``batch_first=False``, or
            :math:`(N, L, D * H_{out})` when ``batch_first=True``. If a
            :class:`torch.nn.utils.rnn.PackedSequence` has been given as the
            input, the output will also be a packed sequence.
        h_n : Tensor, shape = :math:`(D * \text{num\_layers}` or
        :math:`(D * \text{num\_layers}, N, H_{out})`.
            Tensor containing the final hidden state for each element in the
            batch. The shape for unbatched input is
            :math:`(D * \text{num\_layers}, H_{out})`. For batched input,
            the shape is :math:`(D * \text{num\_layers}, N, H_{out})`.
        """
        return super().forward(input, hx)


class DelayLineReservoirESN(ESN):
    r"""
    Applies a multi-layer Echo State Network with :math:`\tanh` or
    :math:`\text{ReLU}` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T)

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used
    instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int, default = 1
        Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
        stacking two RCNs together to form a `deep RCN`, with the second RCN
        taking in outputs of the first RCN and computing the final results.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    forward_weight : Union[float, Tuple[float, ...]], default = 1.
        Scales the forward weights in `w_hh`. If it is a Tuple, it needs to
        have `num_layer` entries for each layer. If it is a number, the value
        in each layer is the same.
    bias_scaling : Union[float, Tuple[float, ...]], default = 0.
        Scales the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_shift : Union[float, Tuple[float, ...]], default = 0.
        Shifts the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    batch_first : bool, default = False
        If ``True``, then the input and output tensors are provided as
        `(batch, seq, feature)` instead of `(seq, batch, feature)`. Note that
        this does not apply to hidden or cell states. See the Inputs/Outputs
        sections below for details.
    bidirectional:  bool, default = False
        If ``True``, becomes a bidirectional RCN.
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
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih_l[k] : torch.Tensor, shape = `(hidden_size, input_size)` or
    `(hidden_size, num_directions * hidden_size)`.
        The input-hidden weights of the k-th layer of shape
        `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
        `(hidden_size, num_directions * hidden_size)`.
    weight_hh_l[k] : torch.Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of the k-th layer of shape
        `(hidden_size, hidden_size)`.
    bias_ih_l[k] : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of the k-th layer of shape
        `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True,
                 input_scaling: Union[float, Tuple[float, ...]] = 1.,
                 forward_weight: Union[float, Tuple[float, ...]] = 1.,
                 bias_scaling: Union[float, Tuple[float, ...]] = 0.,
                 bias_shift: Union[float, Tuple[float, ...]] = 0.,
                 batch_first: bool = False, bidirectional: bool = False,
                 device: Optional[str] = None, dtype: Optional = None):
        self.forward_weight = value_to_tuple(forward_weight, num_layers)

        super().__init__(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, nonlinearity=nonlinearity, bias=bias,
            input_scaling=input_scaling, spectral_radius=0.,
            bias_scaling=bias_scaling, bias_shift=bias_shift,
            input_sparsity=0., recurrent_sparsity=0., batch_first=batch_first,
            bidirectional=bidirectional, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for name, weight in self.named_parameters():
            _, _, layer = name.split("_")
            layer_idx = int(layer[1:])
            weight.requires_grad_(False)
            if "weight_ih" in name:
                init.bernoulli_(
                    weight, p=0.5, std=self.input_scaling[layer_idx])
            elif "weight_hh" in name:
                init.dlr_weights_(
                    weight, forward_weight=self.forward_weight[layer_idx])
            elif "bias_ih" in name:
                init.bernoulli_(
                    weight, -(self.bias_scaling + self.bias_shift)[layer_idx],
                    (self.bias_scaling + self.bias_shift)[layer_idx])
            elif "bias_hh" in name:
                nn.init.zeros_(weight)
            else:
                print(name)


class DelayLineReservoirWithFeedbackESN(ESN):
    r"""
    Applies a multi-layer Echo State Network with :math:`\tanh` or
    :math:`\text{ReLU}` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T)

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used
    instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int, default = 1
        Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
        stacking two RCNs together to form a `deep RCN`, with the second RCN
        taking in outputs of the first RCN and computing the final results.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    forward_weight : Union[float, Tuple[float, ...]], default = 1.
        Scales the forward weights in `w_hh`. If it is a Tuple, it needs to
        have `num_layer` entries for each layer. If it is a number, the value
        in each layer is the same.
    feedback_weight : Union[float, Tuple[float, ...]], default = 1.
        Scales the feedback weights in `w_hh`. If it is a Tuple, it needs to
        have `num_layer` entries for each layer. If it is a number, the value
        in each layer is the same.
    bias_scaling : Union[float, Tuple[float, ...]], default = 0.
        Scales the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_shift : Union[float, Tuple[float, ...]], default = 0.
        Shifts the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    batch_first : bool, default = False
        If ``True``, then the input and output tensors are provided as
        `(batch, seq, feature)` instead of `(seq, batch, feature)`. Note that
        this does not apply to hidden or cell states. See the Inputs/Outputs
        sections below for details.
    bidirectional:  bool, default = False
        If ``True``, becomes a bidirectional RCN.
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
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih_l[k] : torch.Tensor, shape = `(hidden_size, input_size)` or
    `(hidden_size, num_directions * hidden_size)`.
        The input-hidden weights of the k-th layer of shape
        `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
        `(hidden_size, num_directions * hidden_size)`.
    weight_hh_l[k] : torch.Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of the k-th layer of shape
        `(hidden_size, hidden_size)`.
    bias_ih_l[k] : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of the k-th layer of shape
        `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True,
                 input_scaling: Union[float, Tuple[float, ...]] = 1.,
                 forward_weight: Union[float, Tuple[float, ...]] = 1.,
                 feedback_weight: Union[float, Tuple[float, ...]] = 1.,
                 bias_scaling: Union[float, Tuple[float, ...]] = 0.,
                 bias_shift: Union[float, Tuple[float, ...]] = 0.,
                 batch_first: bool = False, bidirectional: bool = False,
                 device: Optional[str] = None, dtype: Optional = None):
        self.forward_weight = value_to_tuple(forward_weight, num_layers)
        self.feedback_weight = value_to_tuple(feedback_weight, num_layers)

        super().__init__(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, nonlinearity=nonlinearity, bias=bias,
            input_scaling=input_scaling, spectral_radius=0.,
            bias_scaling=bias_scaling, bias_shift=bias_shift,
            input_sparsity=0., recurrent_sparsity=0., batch_first=batch_first,
            bidirectional=bidirectional, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for name, weight in self.named_parameters():
            _, _, layer = name.split("_")
            layer_idx = int(layer[1:])
            weight.requires_grad_(False)
            if "weight_ih" in name:
                init.bernoulli_(
                    weight, p=0.5, std=self.input_scaling[layer_idx])
            elif "weight_hh" in name:
                init.dlrb_weights_(
                    weight, forward_weight=self.forward_weight[layer_idx],
                    feedback_weight=self.feedback_weight[layer_idx])
            elif "bias_ih" in name:
                init.bernoulli_(
                    weight, -(self.bias_scaling + self.bias_shift)[layer_idx],
                    (self.bias_scaling + self.bias_shift)[layer_idx])
            elif "bias_hh" in name:
                nn.init.zeros_(weight)
            else:
                print(name)


class SimpleCycleReservoirESN(ESN):
    r"""
    Applies a multi-layer Echo State Network with :math:`\tanh` or
    :math:`\text{ReLU}` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T)

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used
    instead of :math:`\tanh`.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    num_layers : int, default = 1
        Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
        stacking two RCNs together to form a `deep RCN`, with the second RCN
        taking in outputs of the first RCN and computing the final results.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : Union[float, Tuple[float, ...]], default = 1.
        Scales the input weights `w_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    forward_weight : Union[float, Tuple[float, ...]], default = 1.
        Scales the forward weights in `w_hh`. If it is a Tuple, it needs to
        have `num_layer` entries for each layer. If it is a number, the value
        in each layer is the same.
    bias_scaling : Union[float, Tuple[float, ...]], default = 0.
        Scales the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    bias_shift : Union[float, Tuple[float, ...]], default = 0.
        Shifts the bias weights `b_ih`. If it is a Tuple, it needs to have
        `num_layer` entries for each layer. If it is a number, the value in
        each layer is the same.
    batch_first : bool, default = False
        If ``True``, then the input and output tensors are provided as
        `(batch, seq, feature)` instead of `(seq, batch, feature)`. Note that
        this does not apply to hidden or cell states. See the Inputs/Outputs
        sections below for details.
    bidirectional:  bool, default = False
        If ``True``, becomes a bidirectional RCN.
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
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih_l[k] : torch.Tensor, shape = `(hidden_size, input_size)` or
    `(hidden_size, num_directions * hidden_size)`.
        The input-hidden weights of the k-th layer of shape
        `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
        `(hidden_size, num_directions * hidden_size)`.
    weight_hh_l[k] : torch.Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of the k-th layer of shape
        `(hidden_size, hidden_size)`.
    bias_ih_l[k] : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of the k-th layer of shape
        `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True,
                 input_scaling: Union[float, Tuple[float, ...]] = 1.,
                 forward_weight: Union[float, Tuple[float, ...]] = 1.,
                 bias_scaling: Union[float, Tuple[float, ...]] = 0.,
                 bias_shift: Union[float, Tuple[float, ...]] = 0.,
                 batch_first: bool = False, bidirectional: bool = False,
                 device: Optional[str] = None, dtype: Optional = None):
        self.forward_weight = value_to_tuple(forward_weight, num_layers)

        super().__init__(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, nonlinearity=nonlinearity, bias=bias,
            input_scaling=input_scaling, spectral_radius=0.,
            bias_scaling=bias_scaling, bias_shift=bias_shift,
            input_sparsity=0., recurrent_sparsity=0., batch_first=batch_first,
            bidirectional=bidirectional, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for name, weight in self.named_parameters():
            _, _, layer = name.split("_")
            layer_idx = int(layer[1:])
            weight.requires_grad_(False)
            if "weight_ih" in name:
                init.bernoulli_(
                    weight, p=0.5, std=self.input_scaling[layer_idx])
            elif "weight_hh" in name:
                init.scr_weights_(
                    weight, forward_weight=self.forward_weight[layer_idx])
            elif "bias_ih" in name:
                init.bernoulli_(
                    weight, -(self.bias_scaling + self.bias_shift)[layer_idx],
                    (self.bias_scaling + self.bias_shift)[layer_idx])
            elif "bias_hh" in name:
                nn.init.zeros_(weight)
            else:
                print(name)


class ESNCell(nn.RNNCell):
    r"""
    An Echo State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : float, default = 1.
        Scales the input weights `w_ih`.
    spectral_radius : float, default = 1.
        Scales the recurrent weights `w_hh`.
    bias_scaling : float, default = 0.
        Scales the bias weights `b_ih`.
    bias_shift : float, default = 0.
        Shifts the bias weights `b_ih`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    input_sparsity : float, default = 0.9
        Ratio between zero and non-zero values in the input weights `w_ih`.
    recurrent_sparsity : float, default = 0.9
        Ratio between zero and non-zero values in the recurrent weights `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : torch.Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : torch.Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh", input_scaling: float = 1.,
                 spectral_radius: float = 1., bias_scaling: float = 0.,
                 bias_shift: float = 0., leakage: float = 1.,
                 input_sparsity: float = 0.9, recurrent_sparsity: float = 0.9,
                 device: Optional[str] = None, dtype: Optional = None):
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.bias_scaling = bias_scaling
        self.bias_shift = bias_shift
        self.leakage = leakage
        self.input_sparsity = input_sparsity
        self.recurrent_sparsity = recurrent_sparsity
        self.input_sparsity = input_sparsity
        self.recurrent_sparsity = recurrent_sparsity
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias,
            nonlinearity=nonlinearity, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        nn.init.sparse_(self.weight_ih, sparsity=self.input_sparsity,
                        std=self.input_scaling)
        nn.init.sparse_(self.weight_hh, sparsity=self.recurrent_sparsity,
                        std=1.)
        init.spectral_norm_(self.weight_hh)
        self.weight_hh *= self.spectral_radius
        if self.bias:
            nn.init.uniform_(self.bias_ih,
                             -self.bias_scaling + self.bias_shift,
                             self.bias_scaling + self.bias_shift)
            nn.init.zeros_(self.bias_hh)

    def set_external_input_weights(self, weights: torch.Tensor) -> None:
        """
        Set externally initialized input weights for the layer.

        Parameters
        ----------
        weights : torch.Tensor, shape = (out_features, in_features).
            The externally initialized weight tensor.
        """
        if weights.shape != (self.hidden_size, self.input_size):
            raise ValueError(
                f"Shape of weights ({weights.shape}) does not match the "
                f"expected shape ({(self.hidden_size, self.input_size)}).")

        with torch.no_grad():
            self.weight_ih.copy_(weights)
            self.weight_ih.requires_grad_(False)

    def set_external_recurrent_weights(self, weights: torch.Tensor) -> None:
        """
        Set externally initialized recurrent weights for the layer.

        Parameters
        ----------
        weights : torch.Tensor, shape = (out_features, in_features).
            The externally initialized weight tensor.
        """
        if weights.shape != (self.hidden_size, self.hidden_size):
            raise ValueError(
                f"Shape of weights ({weights.shape}) does not match the "
                f"expected shape ({(self.hidden_size, self.hidden_size)}).")

        with torch.no_grad():
            self.weight_hh.copy_(weights)
            self.weight_hh.requires_grad_(False)

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
            self.bias_ih.copy_(bias)
            self.bias_ih.requires_grad_(False)

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        r"""
        Forward function of the neural network. Pass the input features through
        the hidden layers and return the hidden layer states of the final
        layer.

        Parameters
        ----------
        input : torch.Tensor, shape = :math:`(N, H_{in})` or :math:`(H_{in})`.
            Tensor containing the input features. The shape for unbatched
            input is :math:`(H_{in})`. For batched input, the shape is
            :math:`(N, H_{in})`.
        hx : Optional[torch.Tensor], shape :math:`(H_{out})`
             or :math:`(N, H_{out}).
            Tensor containing the initial hidden state. The shape for unbatched
            input is :math:`(H_{out})`. For batched input, the shape is
            :math:`(N, H_{out})`. Defaults to zero if not provided.

        Returns
        -------
        output : Tensor, shape = :math:`(H_{out})` or:math:`(N, H_{out})`.
            Tensor containing the hidden states `h' `of the RNN. The shape for
            unbatched input is :math:`(H_{out})`. For batched input, the shape
            is :math:`(N, H_{out})`.
        """
        if self.leakage < 1.:
            return self._leaky_forward(input=input, hx=hx)
        return super().forward(input=input, hx=hx)

    def _leaky_forward(self, input: torch.Tensor,
                       hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Leaky integration forward function of the neural network. Pass the
        input features through the hidden layers and return the hidden layer
        states of the final layer.

        Parameters
        ----------
        input : torch.Tensor, shape = :math:`(N, H_{in})` or :math:`(H_{in})`.
            Tensor containing the input features. The shape for unbatched
            input is :math:`(H_{in})`. For batched input, the shape is
            :math:`(N, H_{in})`.

        Returns
        -------
        output : torch.Tensor, shape = :math:`(H_{out})` or:math:`(N, H_{out})`
            Tensor containing the hidden states `h' `of the RNN. The shape for
            unbatched input is :math:`(H_{out})`. For batched input, the shape
            is :math:`(N, H_{out})`.
        """
        return \
            (1 - self.leakage) * hx + self.leakage * super().forward(input, hx)


class IdentityESNCell(ESNCell):
    r"""
    An Echo State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    spectral_radius : float, default = 1.
        Scales the recurrent weights `w_hh`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    recurrent_sparsity : float, default = 0.9
        Ratio between zero and non-zero values in the recurrent weights `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : torch.Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, nonlinearity: str = "tanh",
                 spectral_radius: float = 1., leakage: float = 1.,
                 recurrent_sparsity: float = 0.9, device: Optional[str] = None,
                 dtype: Optional = None):
        super().__init__(
            input_size=input_size, hidden_size=input_size, bias=False,
            nonlinearity=nonlinearity, input_scaling=1.,
            spectral_radius=spectral_radius, bias_scaling=0., bias_shift=0.,
            leakage=leakage, input_sparsity=1.,
            recurrent_sparsity=recurrent_sparsity, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        nn.init.eye_(self.weight_ih)
        nn.init.sparse_(self.weight_hh, sparsity=self.recurrent_sparsity,
                        std=1.)
        init.spectral_norm_(self.weight_hh)
        self.weight_hh *= self.spectral_radius
        if self.bias:
            nn.init.uniform_(self.bias_ih,
                             -self.bias_scaling + self.bias_shift,
                             self.bias_scaling + self.bias_shift)
            nn.init.zeros_(self.bias_hh)


class IdentityEuSNCell(ESNCell):
    r"""
    An Euler State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    spectral_radius : float, default = 1.
        Scales the recurrent weights `w_hh`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    recurrent_sparsity : float, default = 0.9
        Ratio between zero and non-zero values in the recurrent weights `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : torch.Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : torch.Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, nonlinearity: str = "tanh",
                 leakage: float = 1., recurrent_sparsity: float = 0.9,
                 recurrent_scaling: float = 1., gamma: float = 0.001,
                 epsilon: float = 0.01, device: Optional[str] = None,
                 dtype: Optional = None):
        self.recurrent_scaling = recurrent_scaling
        self.gamma = gamma
        self.epsilon = epsilon
        super().__init__(
            input_size=input_size, hidden_size=input_size, bias=False,
            nonlinearity=nonlinearity, input_scaling=1.,
            spectral_radius=0., bias_scaling=0., bias_shift=0.,
            leakage=leakage, input_sparsity=1.,
            recurrent_sparsity=recurrent_sparsity, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        nn.init.eye_(self.weight_ih)
        nn.init.sparse_(self.weight_hh, sparsity=self.recurrent_sparsity,
                        std=1.)
        init.antisymmetric_norm_(self.weight_hh)
        init.diffusion_norm_(self.weight_hh, gamma=self.gamma)
        if self.bias:
            nn.init.uniform_(self.bias_ih,
                             -self.bias_scaling + self.bias_shift,
                             self.bias_scaling + self.bias_shift)
            nn.init.zeros_(self.bias_hh)

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        r"""
        Forward function of the neural network. Pass the input features through
        the hidden layers and return the hidden layer states of the final
        layer.

        Parameters
        ----------
        input : torch.Tensor, shape = :math:`(N, H_{in})` or :math:`(H_{in})`.
            Tensor containing the input features. The shape for unbatched
            input is :math:`(H_{in})`. For batched input, the shape is
            :math:`(N, H_{in})`.
        hx : Optional[torch.Tensor], shape :math:`(H_{out})`
             or :math:`(N, H_{out}).
            Tensor containing the initial hidden state. The shape for unbatched
            input is :math:`(H_{out})`. For batched input, the shape is
            :math:`(N, H_{out})`. Defaults to zero if not provided.

        Returns
        -------
        output : Tensor, shape = :math:`(H_{out})` or:math:`(N, H_{out})`.
            Tensor containing the hidden states `h' `of the RNN. The shape for
            unbatched input is :math:`(H_{out})`. For batched input, the shape
            is :math:`(N, H_{out})`.
        """
        return self._euler_forward(input=input, hx=hx)

    def _euler_forward(self, input: torch.Tensor,
                       hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Euler forward function of the neural network. Pass the input features
        through the hidden layers and return the hidden layer states of the
        final layer.

        Parameters
        ----------
        input : torch.Tensor, shape = :math:`(N, H_{in})` or :math:`(H_{in})`.
            Tensor containing the input features. The shape for unbatched
            input is :math:`(H_{in})`. For batched input, the shape is
            :math:`(N, H_{in})`.

        Returns
        -------
        output : torch.Tensor, shape = :math:`(H_{out})` or:math:`(N, H_{out})`
            Tensor containing the hidden states `h' `of the RNN. The shape for
            unbatched input is :math:`(H_{out})`. For batched input, the shape
            is :math:`(N, H_{out})`.
        """
        return hx + self.epsilon * super().forward(input, hx)


class DelayLineReservoirCell(ESNCell):
    r"""
    An Echo State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : float, default = 1.
        Scales the input weights `w_ih`.
    bias_scaling : float, default = 0.
        Scales the bias weights `b_ih`.
    bias_shift : float, default = 0.
        Shifts the bias weights `b_ih`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    forward_weight : float, default = 1.
        Scales the forward weights in `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh", input_scaling: float = 1.,
                 bias_scaling: float = 0., bias_shift: float = 0.,
                 leakage: float = 1., forward_weight: float = .9,
                 device: Optional[str] = None, dtype: Optional = None):
        self.forward_weight = forward_weight
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias,
            nonlinearity=nonlinearity, input_scaling=input_scaling,
            spectral_radius=0., bias_scaling=bias_scaling,
            bias_shift=bias_shift, leakage=leakage, input_sparsity=0.,
            recurrent_sparsity=0., device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        init.bernoulli_(self.weight_ih, p=.5, std=self.input_scaling)
        init.dlr_weights_(self.weight_hh, forward_weight=self.forward_weight)
        if self.bias:
            init.bernoulli_(self.bias_ih, .5, self.bias_scaling)
            nn.init.zeros_(self.bias_hh)


class DelayLineReservoirWithFeedbackCell(ESNCell):
    r"""
    An Echo State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : float, default = 1.
        Scales the input weights `w_ih`.
    bias_scaling : float, default = 0.
        Scales the bias weights `b_ih`.
    bias_shift : float, default = 0.
        Shifts the bias weights `b_ih`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    forward_weight : float, default = .9
        Scales the forward weights in `w_hh`.
    feedback_weight : float, default = .1
        Scales the feedback weights in `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh", input_scaling: float = 1.,
                 bias_scaling: float = 0., bias_shift: float = 0.,
                 leakage: float = 1., forward_weight: float = .9,
                 feedback_weight: float = .1, device: Optional[str] = None,
                 dtype: Optional = None):
        self.forward_weight = forward_weight
        self.feedback_weight = feedback_weight
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias,
            nonlinearity=nonlinearity, input_scaling=input_scaling,
            spectral_radius=0., bias_scaling=bias_scaling,
            bias_shift=bias_shift, leakage=leakage, input_sparsity=0.,
            recurrent_sparsity=0., device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        init.bernoulli_(self.weight_ih, p=.5, std=self.input_scaling)
        init.dlrb_weights_(self.weight_hh, forward_weight=self.forward_weight,
                           feedback_weight=self.feedback_weight)
        if self.bias:
            init.bernoulli_(self.bias_ih, .5, self.bias_scaling)
            nn.init.zeros_(self.bias_hh)


class SimpleCycleReservoirCell(ESNCell):
    r"""
    An Echo State Network cell with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity.

    .. math::
        h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h)

    If :attr:`nonlinearity` is ``'relu'``, then ReLU is used instead of tanh.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input `x`.
    hidden_size : int
        The number of features in the hidden state `h`.
    nonlinearity : str, default = 'tanh'
        The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
    bias : bool, default = True
        If ``False``, then the layer does not use bias weights `b_ih`.
    input_scaling : float, default = 1.
        Scales the input weights `w_ih`.
    bias_scaling : float, default = 0.
        Scales the bias weights `b_ih`.
    bias_shift : float, default = 0.
        Shifts the bias weights `b_ih`.
    leakage : float, default = 1.
        Parameter to determine the degree of leaky integration.
    forward_weight : float, default = 1.
        Scales the forward weights in `w_hh`.
    device: Optional[str], default = None

    dtype: Optional, default = None


    Notes
    -----
    Important sizes:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            L ={} & \text{sequence length} \\
            D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
            H_{in} ={} & \text{input\_size} \\
            H_{out} ={} & \text{hidden\_size}
        \end{aligned}


    Attributes # noqa
    ----------
    weight_ih : Tensor, shape = `(hidden_size, input_size)`.
        The input-hidden weights of shape `(hidden_size, input_size)`.
    weight_hh : Tensor, shape = `(hidden_size, hidden_size)`.
        The hidden-hidden weights of shape `(hidden_size, hidden_size)`.
    bias_ih : Tensor, shape = `(hidden_size)`.
        The learnable input-hidden bias of shape `(hidden_size)`.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh", input_scaling: float = 1.,
                 bias_scaling: float = 0., bias_shift: float = 0.,
                 leakage: float = 1., forward_weight: float = .9,
                 device: Optional[str] = None, dtype: Optional = None):
        self.forward_weight = forward_weight
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias,
            nonlinearity=nonlinearity, input_scaling=input_scaling,
            spectral_radius=0., bias_scaling=bias_scaling,
            bias_shift=bias_shift, leakage=leakage, input_sparsity=0.,
            recurrent_sparsity=0., device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Initialize the weight matrices of the neural network."""
        for weight in self.parameters():
            weight.requires_grad_(False)
        init.bernoulli_(self.weight_ih, p=.5, std=self.input_scaling)
        init.scr_weights_(self.weight_hh, forward_weight=self.forward_weight)
        if self.bias:
            init.bernoulli_(self.bias_ih, .5, self.bias_scaling)
            nn.init.zeros_(self.bias_hh)
