.. _installation guide:

==================
Installation guide
==================

PyRCN requires **Python 3.10 or newer**. It is built on top of widely used
scientific Python packages, most importantly `scikit-learn
<https://scikit-learn.org/stable>`_ (for the estimator API) and `PyTorch
<https://pytorch.org>`_ (for the compute backend), together with numpy, scipy
and pandas. These dependencies are installed automatically.

To find out which Python version you are running, open a terminal (PowerShell or
Command Prompt on Windows, a shell on Linux and macOS) and run:

.. code-block:: bash

    python --version

If this reports a version older than 3.10, please install a newer Python before
continuing. To keep PyRCN and its dependencies isolated from the rest of your
system, we recommend installing it into a virtual environment. The `Python
documentation on virtual environments and packages
<https://docs.python.org/3/tutorial/venv.html>`_ explains how to create one.

Installation using ``pip``
--------------------------

PyRCN is published on the `Python Package Index (PyPI) <https://pypi.org/>`_, so
you can install the latest release with a single command:

.. code-block:: bash

    pip install pyrcn

To upgrade an existing installation to the newest release:

.. code-block:: bash

    pip install --upgrade pyrcn

To check that PyRCN is installed and see its version:

.. code-block:: bash

    pip show pyrcn

Optional dependencies (extras)
------------------------------

Some functionality needs additional packages, grouped into extras that you can
request in square brackets.

* ``test`` installs the tools used to run the test suite (pytest, pytest-cov,
  mypy and flake8):

  .. code-block:: bash

      pip install pyrcn[test]

* ``examples`` installs the packages used by the example scripts and Jupyter
  notebooks (matplotlib, seaborn, ipywidgets, ipympl and tqdm):

  .. code-block:: bash

      pip install pyrcn[examples]

You can combine extras, for example ``pip install pyrcn[test,examples]``.

A note on PyTorch
-----------------

PyRCN depends on PyTorch. By default, ``pip install pyrcn`` pulls in a build of
PyTorch that runs on the CPU, which works on every platform and requires no
further setup.

To run PyRCN on a GPU, install the CUDA build of PyTorch that matches your
system before or after installing PyRCN. PyTorch publishes these builds through
its own package index. For example, for CUDA 12.4:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu124

To force the CPU-only build explicitly:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Always pick the command that matches your operating system and CUDA version from
the official `PyTorch installation selector
<https://pytorch.org/get-started/locally/>`_. Installing the correct PyTorch
build first, then ``pip install pyrcn``, avoids replacing an already working
GPU installation.

In case of any problems, please report bugs at the `issue tracker on GitHub`_ or
send a mail to Peter Steiner `peter.steiner@pyrcn.net
<mailto:peter.steiner@pyrcn.net>`_.

Installation from source
------------------------

Installing from source is recommended if you would like to contribute to PyRCN.
The source code is hosted on `GitHub
<https://github.com/PlasmaControl/PyRCN>`_.

The ``main`` branch holds the latest stable version. To work with the most
recent development version, check out the ``dev`` branch instead. Clone the
repository (or download and unzip it) and install it in editable mode:

.. code-block:: bash

    git clone https://github.com/PlasmaControl/PyRCN.git
    pip install -e PyRCN[test]

.. _issue tracker on GitHub: https://github.com/PlasmaControl/PyRCN/issues
