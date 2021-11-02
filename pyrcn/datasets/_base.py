"""The :mod:`pyrcn.datasets` includes base toy datasets."""

from typing import Union, Tuple
import numpy as np
import collections
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.datasets import load_digits as sklearn_load_digits
from sklearn.utils import Bunch


def _mg_eq(xt: float, xtau: float, a: float = 0.2, b: float = 0.1,
           n: int = 10) -> float:
    """Mackey-Glass time delay diffential equation, at values x(t) and x(t-tau)."""
    return -b*xt + a*xtau / (1+xtau**n)


def _mg_rk4(xt: float, xtau: float, a: float, b: float, n: int,
            h: float = 1.0) -> float:
    """Runge-Kuta method (RK4) for Mackey-Glass timeseries discretization."""
    k1 = h * _mg_eq(xt, xtau, a, b, n)
    k2 = h * _mg_eq(xt + 0.5*k1, xtau, a, b, n)
    k3 = h * _mg_eq(xt + 0.5*k2, xtau, a, b, n)
    k4 = h * _mg_eq(xt + k3, xtau, a, b, n)

    return xt + k1/6 + k2/3 + k3/3 + k4/6


@_deprecate_positional_args
def mackey_glass(n_timesteps: int, n_future: Union[int, np.integer] = 1,
                 tau: Union[int, np.integer] = 17, a: float = 0.2, b: float = 0.1,
                 n: int = 10, x0: float = 1.2, h: float = 1.0,
                 seed: Union[int, np.random.RandomState, None] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Mackey-Glass time-series.

    Mackey-Glass timeseries [#]_ [#]_, computed from the Mackey-Glass
    delayed differential equation:
    .. math::
        \\frac{x}{t} = \\frac{ax(t-\\tau)}{1+x(t-\\tau)^n} - bx(t)

    Parameters
    ----------
        n_timesteps : Union[int, np.integer]
            Number of timesteps to compute.
        n_future : Union[int, np.integer], optional
            distance between input and target samples.
            By default, equal to 1.
        tau : Union[int, np.integer], optional
            Time delay :math:`\\tau` of Mackey-Glass equation.
            By defaults, equal to 17. Other values can
            change the choatic behaviour of the timeseries.
        a : float, optional
            :math:`a` parameter of the equation.
            By default, equal to 0.2.
        b : float, optional
            :math:`b` parameter of the equation.
            By default, equal to 0.1.
        n : Union[int, np.integer], optional
            :math:`n` parameter of the equation.
            By default, equal to 10.
        x0 : float, optional
            Initial condition of the timeseries.
            By default, equal to 1.2.
        h : float, optional
            Time delta for the Runge-Kuta method. Can be assimilated
            to the number of discrete point computed per timestep.
            By default, equal to 1.0.
        seed : Optional[int, np.random.RandomState], default=None
            Random state seed for reproducibility.
    Returns
    -------
        np.ndarray
            Mackey-Glass timeseries.
    Note
    ----
        As Mackey-Glass is defined by delayed time differential equations,
        the first timesteps of the timeseries can't be initialized at 0
        (otherwise, the first steps of computation involving these
        not-computed-yet-timesteps would yield inconsistent results).
        A random number generator is therefore used to produce random
        initial timesteps based on the value of the initial condition
        passed as parameter. A default seed is hard-coded to ensure
        reproducibility in any case.

    References
    ----------
        .. [#] M. C. Mackey and L. Glass, ‘Oscillation and chaos in physiological
               control systems’, Science, vol. 197, no. 4300, pp. 287–289, Jul. 1977,
               doi: 10.1126/science.267326.
        .. [#] `Mackey-Glass equations
                <https://en.wikipedia.org/wiki/Mackey-Glass_equations>`_
                on Wikipedia.
    """
    # a random state is needed as the method used to discretize
    # the timeseries needs to use randomly generated initial steps
    # based on the initial condition passed as parameter.
    if isinstance(seed, np.random.RandomState):
        rs = seed
    elif seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random.RandomState(42)
    # generate random first step based on the value
    # of the initial condition
    history_length = int(np.floor(tau/h))
    history = collections.deque(x0 * np.ones(history_length)
                                + 0.2 * (rs.rand(history_length) - 0.5))
    xt = x0
    X = np.zeros(n_timesteps + 1)
    for i in range(0, n_timesteps):
        X[i] = xt
        if tau == 0:
            xtau = 0.0
        else:
            xtau = history.popleft()
            history.append(xt)
        xth = _mg_rk4(xt, xtau, a=a, b=b, n=n)
        xt = xth

    y = X[1:]
    X = X[:-1]
    return X, y


@_deprecate_positional_args
def load_digits(*, n_class: Union[int, np.integer] = 10,
                return_X_y: bool = False,
                as_frame: bool = False,
                as_sequence: bool = False) -> Union[Bunch, tuple]:
    """
    Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.
    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============
    Read more in the :ref:`User Guide <digits_dataset>`.

    Parameters
    ----------
    n_class : Union[int, np.integer], default=10
        The number of classes to return. Between 0 and 10.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
        .. versionadded:: 0.18
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.
        .. versionadded:: 0.23
    as_frame : bool, default=False
        If True, the data is returned as a sequence in the data format required
        by PyRCN.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
            .. versionadded:: 0.20
        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
            .. versionadded:: 0.23
        images: {ndarray} of shape (1797, 8, 8)
            The raw image data.
        DESCR: str
            The full description of the dataset.
    (data, target) : tuple if ``return_X_y`` is True
    """
    if as_sequence and return_X_y and not as_frame:
        X_ori, y_ori = sklearn_load_digits(n_class=n_class, return_X_y=return_X_y,
                                           as_frame=as_frame)
        X = np.empty(shape=(X_ori.shape[0],), dtype=object)
        y = np.empty(shape=(X_ori.shape[0],), dtype=object)
        for k, (X_single, y_single) in enumerate(zip(X_ori, y_ori)):
            X[k] = X_single.reshape(8, 8).T
            y[k] = np.atleast_1d(y_single)
        return X, y
    else:
        return sklearn_load_digits(n_class=n_class, return_X_y=return_X_y,
                                   as_frame=as_frame)
