Introduction
============

`PyRCN`_ is a light-weight and transparent Python 3 framework that implements ESNs and is based on widely used scientific Python packages, such as numpy or scipy. The API is fully scikit-learn-compatible, so that users of scikit-learn do not need to restructure their research data in order to use ESNs. Interested used can directly use scikit-learns built-in parameter optimization methods and example datasets.

PyRCN is used by the Chair of Speech Technology and Cognitive Systems, Institute for Acoustics and Speech Communications, Technische Universität Dresden, Dresden, Germany (https://tu-dresden.de/ing/elektrotechnik/ias/stks?set_language=en) and IDLab (Internet and Data Lab), Ghent University, Ghent, Belgium (https://www.ugent.be/ea/idlab/en). 

It is an acronym for "Python Reservoir Computing Networks". 

Currently, it implements Echo State Networks (ESNs) by Herbert Jaeger in different flavors, e.g. Classifier and Regressor. It is actively developed to be extended into several directions:

- Incorporate Feedback
- Better sequence handling with sktime (http://sktime.org/)
- A unified API to stack ESNs
- More towards future work: Related architectures, such as Extreme Learning Machines (ELMs) and Liquid State Machines (LSMs)

PyRCN has successfully been used for several tasks:

- Music Information Retrieval (MIR)
    - Multipitch Tracking
    - Onset Detection
- Time Series Prediction
    - Mackey-Glass benchmark test
    - Stock Price Prediction
- Tasks we are working on at the moment:
    - Beat Tracking in music signals
    - Pattern recognition in sensor data
    - Phoneme recognition

Please see the "Reference" section for more information. Code examples to getting started with PyRCN are included in the "examples" directory.

.. _PyRCN: https://github.com/TUD-STKS/PyRCN
