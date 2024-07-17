"""Custom weight initialization methods."""
import torch


def antisymmetric_norm_(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given square matrix to be antisymmetric, i.e., by
    computing weight - weight.T .

    Parameters
    ----------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        Tensor (at least 2D) to be normalized.

    Returns
    -------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        The normalized tensor (at least 2D).
    """
    with torch.no_grad():
        tensor = tensor - tensor.T
    return tensor


def diffusion_norm_(tensor: torch.Tensor, gamma: float) -> torch.Tensor:
    r"""
    Normalize a given square matrix to be antisymmetric, i.e., by
    computing weight - \gamma I .

    Parameters
    ----------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        Tensor (at least 2D) to be normalized.

    Returns
    -------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        The normalized tensor (at least 2D).
    """
    with torch.no_grad():
        tensor = tensor - gamma * torch.eye(tensor.shape[0])
    return tensor


def spectral_norm_(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a given square matrix to a unit spectral radius, i.e., to a
    maximum absolute eigenvalue of 1.

    Parameters
    ----------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        Tensor (at least 2D) to be normalized.

    Returns
    -------
    tensor : torch.Tensor, shape=(hidden_size, hidden_size)
        The normalized tensor (at least 2D).
    """
    eigvals = torch.linalg.eigvals(tensor)
    with torch.no_grad():
        tensor /= eigvals.abs().max()
    return tensor


def dlr_weights_(tensor: torch.Tensor, forward_weight: float = 0.9) -> \
        torch.Tensor:
    r"""
    Fills the 2D input `Tensor` such that the non-zero elements will be a
    delay line, and each non-zero element has exactly the same weight value,
    as described in `Minimum Complexity Echo State Network` - Rodan, A. (2010).

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`.
    forward_weight : float, default = 0.9
        The non-zero weight that is placed in the lower subdiagonal of the
        tensor.

    Returns
    -------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`, in which the lower subdiagonal is
        filled with always the same value.

    Examples
    --------
    >>> w = torch.empty(3, 5)
    >>> dlr_weights_(w, forward_weight=0.9)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape

    with torch.no_grad():
        tensor.zero_()
        for col_idx in range(1, cols):
            tensor[col_idx, col_idx-1] = forward_weight
    return tensor


def dlrb_weights_(tensor, forward_weight: float = 0.9,
                  feedback_weight: float = 0.1) -> torch.Tensor:
    r"""
    Fills the 2D input `Tensor` such that the non-zero elements will be a
    delay line with feedback connections, and each non-zero element has
    exactly the same weight value, as described in
    `Minimum Complexity Echo State Network` - Rodan, A. (2010).

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`.
    forward_weight : float, default = 0.9
        The non-zero weight that is placed in the lower subdiagonal of the
        tensor.
    feedback_weight : float, default = 0.1
        The non-zero weight that is placed in the upper subdiagonal of the
        tensor.

    Returns
    -------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`, in which the lower and upper
        subdiagonals are filled with always the same values.

    Examples
    --------
    >>> w = torch.empty(3, 5)
    >>> dlrb_weights_(w, forward_weight=0.9, feedback_weight=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape

    with torch.no_grad():
        tensor.zero_()
        for col_idx in range(1, cols):
            tensor[col_idx, col_idx-1] = forward_weight
            tensor[col_idx-1, col_idx] = feedback_weight
    return tensor


def scr_weights_(tensor, forward_weight: float = 0.9) -> torch.Tensor:
    r"""
    Fills the 2D input `Tensor` such that the non-zero elements will be a
    cycle, and each non-zero element has exactly the same weight value, as
    described in `Minimum Complexity Echo State Network` - Rodan, A. (2010).

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`.
    forward_weight : float, default = 0.9
        The non-zero weight that is placed in the lower subdiagonal of the
        tensor.

    Returns
    -------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`, in which the lower and upper
        subdiagonals are filled with always the same values.

    Examples
    --------
    >>> w = torch.empty(3, 5)
    >>> scr_weights_(w, forward_weight=0.9)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape

    with torch.no_grad():
        tensor.zero_()
        for col_idx in range(cols):
            tensor[col_idx, col_idx-1] = forward_weight
    return tensor


def bernoulli_(tensor: torch.Tensor, p: float = .5, std: float = 1.) \
        -> torch.Tensor:
    r"""
    Fills the 2D input `Tensor` as a sparse matrix, where the non-zero elements
    will be binary numbers (0 or 1) drawn from a Bernoulli distribution
    :math:`\text{Bernoulli}(\texttt{p})` and scaled,  as described in
    `Minimum Complexity Echo State Network` - Rodan, A. (2010).

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional `torch.Tensor`.
    p : float, default = 0.5
        Probability to be used for drawing the binary random number.The
        fraction of elements in each column to be set to zero
    std : float, defaul = 1.
        The scaling factor for the weight matrix.

    Examples
    --------
    >>> w = torch.empty(3, 5)
    >>> bernoulli_(w, p=.5, std=1.)
    """
    with torch.no_grad():
        tensor.bernoulli_(p)
        tensor *= 2*std
        tensor -= std
    return tensor
