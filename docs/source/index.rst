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

- Interaction with `sktime <https://sktime.org/>`_
- Interaction with `hmmlearn <https://hmmlearn.readthedocs.io/en/stable/>`_
- More towards future work: Related architectures, such as Liquid State Machines (LSMs) and Perturbative Neural Networks (PNNs)

PyRCN has successfully been used for several tasks:

- Music Information Retrieval (MIR)
    - Multipitch tracking
    - Onset detection
    - *f*\ :sub:`0`\  analysis of spoken language
    - GCI detection in raw audio signals
- Time Series Prediction
    - Mackey-Glass benchmark test
    - Stock price prediction
- Ongoing research tasks:
    - Beat tracking in music signals
    - Pattern recognition in sensor data
    - Phoneme recognition
    - Unsupervised pre-training of RCNs and optimization of ESNs

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

If you use PyRCN, please cite the following publication:

.. code-block:: latex

    @misc{steiner2021pyrcn,
          title={PyRCN: A Toolbox for Exploration and Application of Reservoir Computing Networks}, 
          author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone and Peter Birkholz},
          year={2021},
          eprint={2103.04807},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }


Acknowledgements
----------------

This research was funded by the European Social Fund (Application number: 100327771) and co-financed by tax funds based on the budget approved by the members of the Saxon State Parliament, and by Ghent University.

.. image:: _static/img/SMWA_EFRE-ESF_Sachsen_Logokombi_quer_03.jpg
  :height: 90
  :alt: Europäischer Sozialfonds

.. image:: _static/img/Logo_IDLab_White.png
  :height: 70
  :alt: IDLab

.. image:: _static/img/logo_UGent_EN_RGB_2400_color-on-white.png
  :height: 70
  :alt: Ghent University

.. image:: _static/img/Logo-STKS.jpg
  :height: 70
  :alt: Kognitive Systeme und Sprachtechnologie

.. image:: _static/img/TUD_Logo_HKS41_114.png
  :height: 70
  :alt: Ghent University
