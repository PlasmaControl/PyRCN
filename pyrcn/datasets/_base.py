import collections
from typing import Union
import numpy as np

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.datasets import load_digits as sklearn_load_digits

def _mg_eq(xt, xtau, a=0.2, b=0.1, n=10):
    """
    Mackey-Glass time delay diffential equation, at values x(t) and x(t-tau).
    """
    return -b*xt + a*xtau / (1+xtau**n)


def _mg_rk4(xt, xtau, a, b, n, h=1.0):
    """
    Runge-Kuta method (RK4) for Mackey-Glass timeseries discretization.
    """
    k1 = h * _mg_eq(xt, xtau, a, b, n)
    k2 = h * _mg_eq(xt + 0.5*k1, xtau, a, b, n)
    k3 = h * _mg_eq(xt + 0.5*k2, xtau, a, b, n)
    k4 = h * _mg_eq(xt + k3, xtau, a, b, n)

    return xt + k1/6 + k2/3 + k3/3 + k4/6


@_deprecate_positional_args
def mackey_glass(n_timesteps: int,
                 tau: int = 17,
                 a: float = 0.2,
                 b: float = 0.1,
                 n: int = 10,
                 x0: float = 1.2,
                 h: float = 1.0,
                 seed: Union[int, np.random.RandomState] = 5555) -> np.ndarray:
    """Mackey-Glass timeseries [#]_ [#]_, computed from the Mackey-Glass
    delayed differential equation:
    .. math::
        \\frac{x}{t} = \\frac{ax(t-\\tau)}{1+x(t-\\tau)^n} - bx(t)
    Parameters
    ----------
        n_timesteps : int
            Number of timesteps to compute.
        tau : int, optional
            Time delay :math:`\\tau` of Mackey-Glass equation.
            By defaults, equal to 17. Other values can
            change the choatic behaviour of the timeseries.
        a : float, optional
            :math:`a` parameter of the equation.
            By default, equal to 0.2.
        b : float, optional
            :math:`b` parameter of the equation.
            By default, equal to 0.1.
        n : int, optional
            :math:`n` parameter of the equation.
            By default, equal to 10.
        x0 : float, optional
            Initial condition of the timeseries.
            By default, equal to 1.2.
        h : float, optional
            Time delta for the Runge-Kuta method. Can be assimilated
            to the number of discrete point computed per timestep.
            By default, equal to 1.0.
        seed : int or RandomState
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
        reproducibility in any case. It can be changed with the
        :py:func:`reservoirpy.datasets.seed` function.
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
        rs = np.random.RandomState(5555)

    # generate random first step based on the value
    # of the initial condition
    history_length = int(np.floor(tau/h))
    history = collections.deque(x0 * np.ones(history_length) + 0.2 * (rs.rand(history_length) - 0.5))
    xt = x0

    X = np.zeros(n_timesteps)

    for i in range(0, n_timesteps):
        X[i] = xt

        if tau == 0:
            xtau = 0.0
        else:
            xtau = history.popleft()
            history.append(xt)

        xth = _mg_rk4(xt, xtau, a=a, b=b, n=n)

        xt = xth

    return X.reshape(-1, 1)


@_deprecate_positional_args
def load_digits(*, n_class=10, return_X_y=False, as_frame=False, as_sequence=False):
    if as_sequence and return_X_y and not as_frame:
        X_ori, y_ori = sklearn_load_digits(n_class=n_class, return_X_y=return_X_y, as_frame=as_frame)
        X = np.empty(shape=(X_ori.shape[0],), dtype=object)
        y = np.empty(shape=(X_ori.shape[0],), dtype=object)
        for k, (X_single, y_single) in enumerate(zip(X_ori, y_ori)):
            X[k] = X_single.reshape(8, 8).T
            y[k] = np.atleast_1d(y_single)
        return X, y
    else:
        return sklearn_load_digits(n_class=n_class, return_X_y=return_X_y, as_frame=as_frame)
