.. _`pyrcn.nn`:

pyrcn.nn
========

.. automodule:: pyrcn.nn

Reservoir layers
----------------

.. autoclass:: pyrcn.nn.Reservoir
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyrcn.nn.EulerReservoir
   :members:
   :undoc-members:
   :show-inheritance:

Reservoir cells
---------------

.. autoclass:: pyrcn.nn.LeakyESNCell
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyrcn.nn.EulerESNCell
   :members:
   :undoc-members:
   :show-inheritance:

Feature map and readout
------------------------

.. autoclass:: pyrcn.nn.InputFeatureMap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyrcn.nn.IncrementalRidge
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyrcn.nn.LinearReadout
   :members:
   :undoc-members:
   :show-inheritance:

Gradient training
-----------------

.. autofunction:: pyrcn.nn.train_readout

.. autofunction:: pyrcn.nn.torch_generator

Weight initializers
-------------------

.. automodule:: pyrcn.nn.init
   :members:
   :undoc-members:
