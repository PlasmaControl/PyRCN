===============
Getting started
===============

Before going further, please make sure that you have installed **PyRCN** as recommended
in the :ref:`installation guide` in a new virtual environment. Afterwards, you can
continue with this getting started guide.

PyRCN is an open-source project which aims to provide a framework for developing
Reservoir Computing Networks (RCNs) easily and transparently.

To learn more about the theoretical aspects of reservoir computing, you can read the
page :ref:`whats rc`.

We also recommend you to have a look at the introduction page of `ReservoirPy
<https://reservoirpy.readthedocs.io/en/latest/whats_rc.html>`_, which is another great
resource, in particular for Echo State Networks.

Playing with blocks -- Building blocks of Reservoir Computing
=============================================================

In the recent years, several groups have built toolboxes for Reservoir Computing.
However, they were mostly able to implement one specific type of RCNs from
- Echo State Networks
- Extreme Learning Machines
- Liquid State Machines

Problem is that, despite the similarities of different RCN architectures, nobody has
decomposed RCNs in building blocks. Together with PyRCN, we aim to do that and provide
building blocks with which almost any RCN structure can be composed.

Building your first Reservoir Computing Network
-----------------------------------------------

    Essentially, with only one command, an Echo State Network can be defined
    using the :py:class:`pyrcn.echo_state_network.ESNRegressor` or
    :py:class:`pyrcn.echo_state_network.ESNClassifier` class:

    .. doctest::

        >>> from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
        >>> esn = ESNRegressor()
        >>> esn
        ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(),
                     regressor=IncrementalRegression())

    As we can see, the  ``esn`` consists of different building blocks, e.g.
    :py:class:`pyrcn.base.blocks.InputToNode`, :py:class:`pyrcn.base.blocks
    .NodeToNode` and :py:class:`pyrcn.linear_model.IncrementalRegression`.

    The first block is used to connect the input features to the hidden neurons,
    the second building block defines how the connections inside the
    hidden neurons are organized. By default, all connections are randomly initialized
    and fixed.

    In case one would like to customize the building blocks, you can have a look at the
    included modules of :py:module:``pyrcn.base.blocks``.

    .. doctest::

        >>> import pyrcn.base.blocks as blocks
        >>> from inspect import getmembers, isclass
        >>> getmembers(blocks, isclass)
        [('BatchIntrinsicPlasticity', <class 'pyrcn.base.blocks._input_to_node.BatchIntrinsicPlasticity'>),
        ('HebbianNodeToNode', <class 'pyrcn.base.blocks._node_to_node.HebbianNodeToNode'>),
        ('InputToNode', <class 'pyrcn.base.blocks._input_to_node.InputToNode'>),
        ('NodeToNode', <class 'pyrcn.base.blocks._node_to_node.NodeToNode'>),
        ('PredefinedWeightsInputToNode', <class 'pyrcn.base.blocks._input_to_node.PredefinedWeightsInputToNode'>),
        ('PredefinedWeightsNodeToNode', <class 'pyrcn.base.blocks._node_to_node.PredefinedWeightsNodeToNode'>)]

    Obviously, there are a lot of derived modules from the basic building blocks available.
    Look their functions up in the documentation or in examples!

Training a RCN
--------------

    RCNs can be trained on different kinds of data. In particular, ESNs can then be
    trained on sequential data, such as timeseries, especially chaotic ones. In PyRCN,
    we have re-implemented the Mackey-Glass time-series from `ReservoirPy
    <https://reservoirpy.readthedocs.io/en/latest/whats_rc.html>`_, which is a common
    demonstration for ESNs:

    .. doctest::

        >>> from pyrcn.datasets import mackey_glass
        >>> X, y = mackey_glass(n_timesteps=8000)

    If we visualize the Mackey-Glass time-series, we can see that it is a
    quasi-periodic time-series.
    We now use an :py:class:``pyrcn.echo_state_network.ESNRegressor`` to do a one-step
    ahead prediction of this time-series.

    .. image:: _static/img/getting_started_mackey_glass.svg

    To train the ESN, only three steps are required:

    1. Randomly distribute the time-series to each reservoir neuron (**Input-to-Node**).
    2. Compute the state of each neuron based on the current input and the previous
    state.
    2. Compute a linear regression between the reservoir states and the target output.

    These steps are handled via :py:func:`pyrcn.echo_state_network.ESNRegressor.fit`,
    which is the most important function to train the ESN model:

    .. doctest::

        >>> # Fit the ESN model
        >>> esn.fit(X[:4000].reshape(-1, 1), y[:4000])
        ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(),
             regressor=IncrementalRegression(), requires_sequence=False)

    You can see that the ESN can be fitted using only one command. Afterwards, it is
    ready to use!

Testing and predict using the ESN
---------------------------------

    Finally, we use the :py:func:`pyrcn.echo_state_network.ESNRegressor.predict` function
    to use the trained ESN to predict the test data:

    .. doctest::

        >>> y_pred = esn.predict(X[:4000])

    .. image:: _static/img/getting_started_mackey_glass_predicted.svg
