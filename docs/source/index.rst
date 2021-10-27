.. PyRCN documentation master file, created by
   sphinx-quickstart on Tue Oct 26 11:53:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
PyRCN
=====

**A Python 3 framework for building Reservoir Computing Networks (RCNs).**

.. image:: https://badge.fury.io/py/PyRCN.svg
    :target: https://badge.fury.io/py/PyRCN



PyRCN ("Python Reservoir Computing Networks") is a light-weight and transparent Python 3 framework for Reservoir Computing and is based on widely used scientific Python packages, such as numpy or scipy. 
The API is fully `scikit-learn <https://scikit-learn.org/stable>`_-compatible, so that users of scikit-learn do not need to refactor their code in order to use the estimators implemented by this framework. 
Scikit-learn's built-in parameter optimization methods and example datasets can also be used in the usual way.
PyRCN is used by the `Chair of Speech Technology and Cognitive Systems, Institute for Acoustics and Speech Communications, Technische Universität Dresden, Dresden, Germany <https://tu-dresden.de/ing/elektrotechnik/ias/stks?set_language=en>`_
and `IDLab (Internet and Data Lab), Ghent University, Ghent, Belgium <https://www.ugent.be/ea/idlab/en>`_

Currently, it implements Echo State Networks (ESNs) by Herbert Jaeger and Extreme Learning Machines (ELMs) by Guang-Bin Huang in different flavors, e.g. Classifier and Regressor. It is actively developed to be extended into several directions:

* Interaction with `sktime <https://sktime.org/>`_
* Interaction with `hmmlearn <https://hmmlearn.readthedocs.io/en/stable/>`_
* More towards future work: Related architectures, such as Liquid State Machines (LSMs) and Perturbative Neural Networks (PNNs)

PyRCN has successfully been used for several tasks:

* Music Information Retrieval (MIR)
  * Multipitch tracking
  * Onset detection
  * *f*\ :sub:`0`\  analysis of spoken language
  * GCI detection in raw audio signals
* Time Series Prediction
  * Mackey-Glass benchmark test
  * Stock price prediction
* Ongoing research tasks:
  * Beat tracking in music signals
  * Pattern recognition in sensor data
  * Phoneme recognition
  * Unsupervised pre-training of RCNs and optimization of ESNs
