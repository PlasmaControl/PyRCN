.. _installation guide:

==================
Installation guide
==================

PyRCN runs on many different systems.

Before installing PyRCN, make sure that you have a compatible Python distribution installed
on your computer. **PyRCN is meant to be used only with Python 3.7 and higher**.

To check the version of your Python distribution, you can run the following command in a terminal,
in Linux/MacOS/Windows :

.. code-block:: bash

    python --version

We recommend using a virtual environment to avoid any unintended interactions with the dependencies of 
packages that are already installed on your system. 

To learn more about virtual environment, you can check `Python documentation on virual
environments and packages <https://docs.python.org/3/tutorial/venv.html>`_, or the documentation of the
`conda environment manager <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
if you are using Anaconda.

Installation using `pip`
------------------------

PyRCN is hosted by `PyPI <https://pypi.org/project/pyrcn/>`_ and can therefore be installed using `pip` 
on Linux/MacOS/Windows.

To install PyRCN using `pip`, simply run the following command in a terminal:

.. code-block:: bash

    pip install pyrcn

To upgrade an existing version of PyRCN using `pip`, simply run the following command in a terminal:

.. code-block:: bash

    pip install --upgrade pyrcn


To check your installation of ReservoirPy, run:

.. code-block:: bash

    pip show pyrcn

Installation from source
------------------------

You can find the source code of PyRCN on `GitHub <https://github.com/TUD-STKS/PyRCN>`_.

Download the latest version on the ``master`` branch, or any other branch you would like
to install (``dev`` branch or older versions branches). You can also fork the project from
GitHub.

Then, unzip the project (or clone the forked repository). You can then install ReservoirPy in
editable mode using `pip` :

.. code-block:: bash

    pip install -e /path/to/pyrcn
