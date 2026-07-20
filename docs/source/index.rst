.. PyRCN documentation master file, created by
   sphinx-quickstart on Tue Oct 26 11:53:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
PyRCN
=====

**A Python 3 framework for building Reservoir Computing Networks (RCNs).**

|pypi| |license| |ci| |docs| |coverage|

PyRCN ("Python Reservoir Computing Networks") is a light-weight and transparent
Python 3 framework for Reservoir Computing and is based on widely used scientific
Python packages, such as numpy, scipy, and torch.

The API is fully `scikit-learn <https://scikit-learn.org/stable>`_-compatible, so
that users of scikit-learn do not need to refactor their code in order to use the
estimators implemented by this framework. Scikit-learn's built-in parameter
optimization methods and example datasets can also be used in the usual way.

PyRCN is developed and maintained by Peter Steiner, and used by the `Chair of
Speech Technology and Cognitive Systems, Technische Universität Dresden, Germany
<https://tu-dresden.de/ing/elektrotechnik/ias/stks?set_language=en>`_, `IDLab
(Internet and Data Lab), Ghent University, Belgium
<https://www.ugent.be/ea/idlab/en>`_ and the `Plasma Control group
<https://control.princeton.edu/>`_ (Egemen Kolemen) at Princeton University, USA.

Highlights
==========

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: scikit-learn compatible
      :link: getting_started
      :link-type: doc

      Every estimator follows the scikit-learn API. Pipelines, grid search and
      cross-validation work unchanged, with no code to refactor.

   .. grid-item-card:: Echo State Networks & ELMs
      :link: api/pyrcn.echo_state_network
      :link-type: doc

      Echo State Network and Extreme Learning Machine regressors and classifiers,
      composed from reusable building blocks: input-to-node, node-to-node and the
      readout.

   .. grid-item-card:: PyTorch backend
      :link: api/pyrcn.nn
      :link-type: doc

      An optional torch-native backend (:py:mod:`pyrcn.nn`) runs reservoirs,
      input maps and readouts as ``torch.nn`` modules.

   .. grid-item-card:: Flexible training
      :link: api/pyrcn.echo_state_network
      :link-type: doc

      Train the readout in closed form (ridge regression) or iteratively with
      gradient-based optimizers, and optionally make the input and reservoir
      weights trainable end to end.

Applications
============

PyRCN has successfully been used for several tasks:

* Music Information Retrieval (MIR)

  * Multipitch tracking

  * Onset detection

  * *f*\ :sub:`0`\  analysis of spoken language

  * GCI detection in raw audio signals

* Time Series Prediction

  * Mackey-Glass benchmark test

  * Stock price prediction

It is actively used and extended in ongoing research:

* Audio signal processing

* Nuclear fusion research, including plasma diagnostics and event prediction

* Fundamental Reservoir Computing research, from unsupervised pre-training of
  reservoirs to making Reservoir Computing faster and more efficient

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   introduction
   getting_started
   tutorial
   development
   citation
   api/api

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

If you use PyRCN, please cite the following publication:

.. code-block:: bibtex

    @article{Steiner2022pyrcn,
        title = {PyRCN: A toolbox for exploration and application of Reservoir Computing Networks},
        journal = {Engineering Applications of Artificial Intelligence},
        volume = {113},
        pages = {104964},
        year = {2022},
        issn = {0952-1976},
        doi = {10.1016/j.engappai.2022.104964},
        url = {https://www.sciencedirect.com/science/article/pii/S0952197622001713},
        author = {Steiner, Peter and Jalalvand, Azarakhsh and Stone, Simon and Birkholz, Peter},
    }

Acknowledgements
================

The initial development of PyRCN was funded by the European Social Fund
(Application number: 100327771) and co-financed by tax funds based on the
budget approved by the members of the Saxon State Parliament, and by Ghent
University.

The current work is supported by the U.S. Department of Energy, Office of
Science, Office of Fusion Energy Sciences, under Award Nos. DE-FC02-04ER54698
and DE-SC0024527; by the Princeton Laboratory for Artificial Intelligence under
Award No. 2025-97; and by the U.S. Department of Energy under Contract No.
DE-AC02-09CH11466.

.. raw:: html

   <div class="affiliation-logos">
     <img src="_static/img/plasma_control_logo.svg" alt="Plasma Control group, Princeton University">
     <img src="_static/img/Logo-STKS.jpg" alt="Chair of Speech Technology and Cognitive Systems, TU Dresden">
     <img src="_static/img/TUD_Logo_HKS41_114.png" alt="Technische Universität Dresden">
     <img src="_static/img/logo_UGent_EN_RGB_2400_color-on-white.png" alt="Ghent University">
     <img src="_static/img/SMWA_EFRE-ESF_Sachsen_Logokombi_quer_03.jpg" alt="European Social Fund">
   </div>

.. |pypi| image:: https://img.shields.io/pypi/v/pyrcn.svg
   :target: https://pypi.org/project/pyrcn/
   :alt: PyPI version

.. |license| image:: https://img.shields.io/pypi/l/pyrcn.svg
   :target: https://github.com/PlasmaControl/PyRCN/blob/main/LICENSE
   :alt: License

.. |ci| image:: https://github.com/PlasmaControl/PyRCN/actions/workflows/python-test-push.yml/badge.svg
   :target: https://github.com/PlasmaControl/PyRCN/actions/workflows/python-test-push.yml
   :alt: CI status

.. |docs| image:: https://readthedocs.org/projects/pyrcn/badge/?version=latest
   :target: https://pyrcn.readthedocs.io/en/latest/
   :alt: Documentation status

.. |coverage| image:: _static/img/coverage.svg
   :target: https://github.com/PlasmaControl/PyRCN/actions/workflows/python-test-push.yml
   :alt: Test coverage
