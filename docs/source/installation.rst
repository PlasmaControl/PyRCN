Installation
============

Please  either install :ref:`from package <install_from_package>` or :ref:`from source <install_from_source>` in case you would like to contribute to the package. Please have a look at the :ref:`prerequisites <install_prerequisites>` before installing the package.

.. _install_prerequisites:

Prerequisites
-------------

PyRCn is developed using Python 3.6 or newer. It depends on the following packages:

- [numpy>=1.18.1](https://numpy.org/)
- [scipy>=1.2.0](https://scipy.org/)
- [scikit-learn>=0.22.1](https://scikit-learn.org/stable/)
- [joblib>=0.13.2](https://joblib.readthedocs.io)

.. _install_from_package:

Installation from PyPI
----------------------

The easiest and recommended way to install ``PyRCN`` is to use ``pip`` from (PyPI)[https://pypi.org] :

```python
pip install pyrcn   
```

This will install all dependencies automatically.

.. _install_from_source:

Installation from source
------------------------

If you plan to contribute to ``PyRCN``, you can also install the package from source.

Clone the Git repository:

```
git clone https://github.com/TUD-STKS/PyRCN.git
```

Install the package using ``setup.py``:

```
python setup.py install --user
```
