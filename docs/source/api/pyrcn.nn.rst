.. _`pyrcn.nn`:

pyrcn.nn
========

.. automodule:: pyrcn.nn

.. currentmodule:: pyrcn.nn

Reservoir layers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reservoir
   EulerReservoir

Reservoir cells
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LeakyESNCell
   EulerESNCell

Feature map and readout
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   InputFeatureMap
   IncrementalRidge
   LinearReadout

Gradient training
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   train_readout
   torch_generator

Weight initializers
-------------------

.. currentmodule:: pyrcn.nn.init

.. autosummary::
   :toctree: generated
   :nosignatures:

   antisymmetric_recurrent_weights
   bernoulli_input_weights
   cycle_reservoir_with_jumps_weights
   delay_line_feedback_weights
   delay_line_weights
   multi_ring_weights
   normal_recurrent_weights
   pi_digit_input_weights
   simple_cycle_weights
   spectral_normalize
   uniform_bias_weights
   uniform_input_weights
