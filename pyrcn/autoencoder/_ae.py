"""
The :mod:`echo_state_network` contains the ESNRegressor and the ESNClassifier
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>, Azarakhsh Jalalvand <azarakhsh.jalalvand@ugent.be>
# License: BSD 3 clause

import sys
if sys.version_info >= (3, 8):
    from typing import Union, Literal
else:
    from typing_extensions import literal
    from typing import Union

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.extmath import safe_sparse_dot
from pyrcn.base import ACTIVATIONS


class MLPAutoEncoder(MLPRegressor, TransformerMixin):
    """
    Autoencoder based on Multilayer Perceptron.

    This model optimizes the squared error loss function using LBFGS or stochastic gradient descent.

    Parameters
    ----------
    transform_type : Literal['full', 'only_encode', 'only_decode'], default = 'full'
        Which kind of transform to be performed.
        - 'full', entire autoencoder
          returns the encoded input
        - 'only_encode', only encoder part
          returns bottleneck features
        - 'only_decode', only decoder part
          returns the encoded input from an arbitrary encoder
    discard_unused : bool, default = False
        remove unused part (either encoder or decoder).
    bottleneck_index : Union[int, np.integer], default = 1
        index of the bottleneck. This allows to build asymmetric autoencoders.
    hidden_layer_sizes : tuple, length = n_layers - 2, default = (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    activation : Literal['identity', 'logistic', 'tanh', 'relu'], default = 'relu'
        Activation function for the hidden layer.
        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x
        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)
    solver : Literal['lbfgs', 'sgd', 'adam'], default = 'adam'
        The solver for weight optimization.
        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba
        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.
    alpha : float, default = 0.0001
        L2 penalty (regularization term) parameter.
    batch_size : Union[int, Literal['auto']], default = 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`
    learning_rate : Literal['constant', 'invscaling', 'adaptive'], default = 'constant'
        Learning rate schedule for weight updates.
        - 'constant' is a constant learning rate given by
          'learning_rate_init'.
        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)
        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.
        Only used when solver='sgd'.
    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.
    power_t : float, default = 0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.
    max_iter : Union[int, np.integer], default = 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.
    shuffle : bool, default = True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.
    random_state : Union[None, int, np.random.RandomState], RandomState instance, default = None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    tol : float, default = 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.
    verbose : bool, default = False
        Whether to print progress messages to stdout.
    warm_start : bool, default = False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    momentum : float, default = 0.9
        Momentum for gradient descent update.  Should be between 0 and 1. Only
        used when solver='sgd'.
    nesterovs_momentum : bool, default = True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.
    early_stopping : bool, default = False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'
    validation_fraction : float, default = 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True
    beta_1 : float, default = 0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'
    beta_2 : float, default = 0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'
    epsilon : float, default = 1e-8
        Value for numerical stability in adam. Only used when solver='adam'
    n_iter_no_change : Union[int, np.integer], default = 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'
        .. versionadded:: 0.20
    max_fun : Union[int, np.integer], default = 15000
        Only used when solver='lbfgs'. Maximum number of function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of function calls.
        Note that number of function calls will be greater than or equal to
        the number of iterations for the MLPAutoencoder.
        .. versionadded:: 0.22

    Attributes
    ----------
    loss_ : float
        The current loss computed with the loss function.
    best_loss_ : float
        The minimum loss reached by the solver throughout fitting.
    loss_curve_ : list of shape (`n_iter_`,)
        Loss value evaluated at the end of each training step.
        The ith element in the list represents the loss at the ith iteration.
    t_ : int
        The number of training samples seen by the solver during fitting.
        Mathematically equals `n_iters * X.shape[0]`, it means
        `time_step` and it is used by optimizer's learning rate scheduler.
    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    n_iter_ : int
        The number of iterations the solver has run.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : str
        Name of the output activation function.

    Notes
    -----
    MLPAutoencoder trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.
    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.
    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.
    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.
    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).
    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
    @_deprecate_positional_args
    def __init__(self, *, transform_type: Literal['full', 'only_encode', 'only_decode'] = 'full', 
                 discard_unused: bool = False,
                 bottleneck_index: Union[int, np.integer] = 1,
                 hidden_layer_sizes: tuple = (100,), 
                 activation: Literal['identity', 'logistic', 'tanh', 'relu'] = "relu",
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 alpha: float = 0.0001,
                 batch_size: Union[int, Literal['auto']] = 'auto', 
                 learning_rate: Literal['constant', 'invscaling', 'adaptive'] = "constant", 
                 learning_rate_init: float = 0.001,
                 power_t: float = 0.5, 
                 max_iter: Union[int, np.integer]=200, 
                 shuffle: bool=True,
                 random_state: Union[None, int, np.random.RandomState] = None,
                 tol: float = 1e-4, 
                 verbose: bool = False, 
                 warm_start: bool = False, 
                 momentum: float = 0.9, 
                 nesterovs_momentum: bool = True, 
                 early_stopping: bool = False, 
                 validation_fraction: float = 0.1, 
                 beta_1: float = 0.9, 
                 beta_2: float = 0.999, 
                 epsilon: float = 1e-8, 
                 n_iter_no_change: Union[int, np.integer] = 10,
                 max_fun: Union[int, np.integer] = 15000):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation, solver=solver, alpha=alpha,
                         batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init, power_t=power_t,
                         max_iter=max_iter, shuffle=shuffle, random_state=random_state, 
                         tol=tol, verbose=verbose, warm_start=warm_start, 
                         momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                         early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        self.transform_type = transform_type
        self.discard_unused = discard_unused
        self.bottleneck_index = bottleneck_index

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the model to data matrix X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : returns a trained AE model.
        """
        return super().fit(X=X, y=X)
        if self.discard_unused and self.transform_type == 'only_encode':
            self.transformer_weights_ = self.coefs_[:self.bottleneck_index]
            self.coefs_ = self.coefs_[:self.bottleneck_index]
        elif self.discard_unused and self.transform_type == 'only_decode':
            self.intercepts_ = self.intercepts_[self.bottleneck_index:]
            self.intercepts_ = self.intercepts_[self.bottleneck_index:]
        return self

    @property
    def partial_fit(self):
        """
        Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : returns a trained AE model.
        """
        return super()._partial_fit(X=X, y=X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the multi-layer perceptron autoencoder model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            The predicted values.
        """
        return super().predict(X=X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input matrix X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y: ndarray of size (n_samples, n_features)
        """
        if self.transform_type == 'full':
            return self.predict(X=X)
        else:
            # Initialize first layer
            activation = X
            # Forward propagate
            hidden_activation = ACTIVATIONS[self.activation]
            for i in range(self.bottleneck_index):
                activation = safe_sparse_dot(activation, self.coefs_[i])
                activation += self.intercepts_[i]
                hidden_activation(activation)

            return activation
