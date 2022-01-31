"""The :mod:`pyrcn.datasets` includes base toy datasets."""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>
# License: BSD 3 clause

from typing import Union, Tuple, Callable, Any, List, Dict
import numpy as np
from scipy.integrate import solve_ivp
import collections
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.datasets import load_digits as sklearn_load_digits
from sklearn.utils import Bunch


def _mg_eq(x_t: float, x_tau: float, beta: float = 0.2, gamma: float = 0.1,
           n: int = 10) -> float:
    """
    Mackey-Glass time delay diffential equation, at values x(t) and x(t-tau).

    This code was taken from the ReservoirPy library [#]_.

    References
    ----------
    .. [#] Trouvain et al., ‘ ReservoirPy: an Efficient and User-Friendly
           Library to Design Echo State Networks’,  In International Conference
           on Artificial Neural Networks (pp. 494-505). Springer, Cham.
    """
    return beta*x_tau / (1+x_tau**n) - gamma*x_t


def _runge_kutta(equation: Callable, x_t: float, h: float = 1.,
                 **kwargs: Any) -> float:
    # x_t: float,x_tau: float,beta: float,gamma: float,n: int,h: float = 1.0
    """
    General Runge-Kutta method (RK4) for Mackey-Glass timeseries
    discretization.

    This code was taken from the ReservoirPy library [#]_.

    References
    ----------
    .. [#] Trouvain et al., ‘ ReservoirPy: an Efficient and User-Friendly
           Library to Design Echo State Networks’,  In International Conference
           on Artificial Neural Networks (pp. 494-505). Springer, Cham.
    """
    k_1 = equation(x_t, **kwargs)
    k_2 = equation(x_t + 0.5 * k_1, **kwargs)
    k_3 = equation(x_t + 0.5 * k_2, **kwargs)
    k_4 = equation(x_t + k_3, **kwargs)

    return x_t + h*(k_1 + 2*k_2 + 2*k_3 + k_4) / 6


@_deprecate_positional_args
def mackey_glass(n_timesteps: int, n_future: int = 1, tau: int = 17,
                 beta: float = 0.2, gamma: float = 0.1, n: int = 10,
                 x_0: float = 1.2, h: float = 1.0,
                 random_state: Union[int, np.random.RandomState, None] = 42) \
        -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Mackey-Glass time-series.

    Mackey-Glass timeseries [#]_ [#]_, computed from the Mackey-Glass
    delayed differential equation:
    .. math::
        \\frac{dx}{dt} = \\beta\\frac{x(t-\\tau)}{1+x(t-\\tau)^n}-\\gamma x(t)

    Parameters
    ----------
        n_timesteps : int
            Number of timesteps to compute.
        n_future : int, default = 1
            distance between input and target samples.
        tau : int, default = 17
            Time delay :math:`\\tau` of the Mackey-Glass equation. Other
            values can strongly change the chaotic behaviour of the timeseries.
        beta : float, default = 0.2
            :math:`\\beta` parameter of the equation.
        gamma : float, default = 0.1
            :math:`\\gamma` parameter of the equation.
        n : int, default = 10
            :math:`n` parameter of the equation.
        x_0 : float, default = 1.2
            Initial condition of the timeseries.
        h : float, default = 1.0
            Discretization step for the Runge-Kutta method. Can be assimilated
            to the number of discrete point computed per timestep.
        random_state : Union[int, np.random.RandomState, None], default=42
            Random state seed for reproducibility.

    Returns
    -------
        np.ndarray
            Mackey-Glass timeseries.
    Note
    ----
        This code was inspired and adapted from the ReservoirPy library [#]_.

    References
    ----------
        .. [#] M. C. Mackey and L. Glass, ‘Oscillation and chaos in
               physiological control systems’, Science, vol. 197, no. 4300,
               pp. 287–289, Jul. 1977, doi: 10.1126/science.267326.
        .. [#] 'Mackey-Glass equation
                <http://www.scholarpedia.org/article/Mackey-Glass_equation>'_
                on Scholarpedia.
        .. [#] Trouvain et al., 'ReservoirPy: an Efficient and User-Friendly
               Library to Design Echo State Networks',  In International
               Conference on Artificial Neural Networks (pp. 494-505).
               Springer, Cham.
    """
    # a random state is needed as the method used to discretize the
    # timeseries needs to use randomly generated initial steps based on the
    # initial condition passed as parameter.
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state
    elif random_state is not None:
        random_state = np.random.RandomState(random_state)
    else:
        random_state = np.random.RandomState()
    # generate random first step based on the value of the initial condition
    buffer_length = int(np.floor(tau/h))
    buffer = collections.deque(x_0 * np.ones(buffer_length) + 0.2 * (
            random_state.rand(buffer_length) - 0.5))
    x_t = x_0
    x = np.zeros(n_timesteps + n_future)
    for i in range(0, n_timesteps):
        x[i] = x_t
        if tau == 0:
            x_tau = 0.0
        else:
            x_tau = buffer.popleft()
            buffer.append(x_t)
        x_th = _runge_kutta(equation=_mg_eq, x_t=x_t, h=h,
                            x_tau=x_tau, beta=beta, gamma=gamma, n=n)
        x_t = x_th
    return x[:-n_future], x[n_future:]


@_deprecate_positional_args
def lorenz(n_timesteps: int, n_future: int = 1, sigma: float = 10.,
           rho: float = 28., beta: float = 8./3.,
           x_0: Union[List, np.ndarray] = [1.0, 1.0, 1.0], h: float = 0.03,
           **kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Lorenz time-series.

    Lorenz timeseries [#]_ [#]_, computed from the Lorenz delayed differential
    equation:
    .. math::
        \\frac{dx}{dt} = \\sigma(y - x)
        \\frac{dy}{dt} = x(\\rho - z) - y
        \\frac{dz}{dt} = xy - \\beta z

    Parameters
    ----------
        n_timesteps : int
            Number of timesteps to compute.
        n_future : int, default = 1
            distance between input and target samples.
        sigma : float, default = 10
            :math:`\\sigma` parameter of the system.
        rho : float, default = 28.
            :math:`\\rho` parameter of the equation.
        beta : float, default = :math: `\\frac{8}{3}`
            :math:`\\beta` parameter of the equation.
        x_0 : Union[List, np.ndarray], default = [1.0, 1.0, 1.0]
            Initial condition of the timeseries.
        h : float, default = 0.03
            Discretization step for the Runge-Kutta method. Can be assimilated
            to the number of discrete point computed per timestep.

    Returns
    -------
        np.ndarray
            Lorenz attractor timeseries.
    Note
    ----
        This code was inspired and adapted from the ReservoirPy library [#]_.

    References
    ----------
        .. [#] E. N. Lorenz, ‘Deterministic Nonperiodic Flow’,
               Journal of the Atmospheric Sciences, vol. 20, no. 2,
               pp. 130–141, Mar. 1963,
               doi: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.
        .. [#] 'Lorenz system <https://en.wikipedia.org/wiki/Lorenz_system>'_
               on Wikipedia.
        .. [#] Trouvain et al., 'ReservoirPy: an Efficient and User-Friendly
               Library to Design Echo State Networks',  In International
               Conference on Artificial Neural Networks (pp. 494-505).
               Springer, Cham.
    """
    timesteps = np.arange(0., (n_timesteps + n_future) * h, h)

    def lorenz_differential_equation(t: int,
                                     state: Tuple[float, float, float]) \
            -> Tuple[float, float, float]:
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z)
        dz_dt = x * y - beta * z
        return dx_dt, dy_dt, dz_dt

    lorenz_solution = solve_ivp(fun=lorenz_differential_equation,
                                t_span=(0.0, (n_timesteps + n_future) * h),
                                y0=x_0, t_eval=timesteps, **kwargs)
    return (
        lorenz_solution.y.T[:-n_future, :], lorenz_solution.y.T[n_future:, :])


@_deprecate_positional_args
def load_digits(*, n_class: Union[int, np.integer] = 10,
                return_X_y: bool = False, as_frame: bool = False,
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
        X_ori, y_ori = sklearn_load_digits(
            n_class=n_class, return_X_y=return_X_y, as_frame=as_frame)
        X = np.empty(shape=(X_ori.shape[0],), dtype=object)
        y = np.empty(shape=(X_ori.shape[0],), dtype=object)
        for k, (X_single, y_single) in enumerate(zip(X_ori, y_ori)):
            X[k] = X_single.reshape(8, 8).T
            y[k] = np.atleast_1d(y_single)
        return X, y
    else:
        return sklearn_load_digits(
            n_class=n_class, return_X_y=return_X_y, as_frame=as_frame)
