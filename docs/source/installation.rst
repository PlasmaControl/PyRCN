==================
Installation guide
==================

Before installing PyRCN, make sure that you have at least **Python 3.7** installed on
your system. **PyRCN is developed and supposed to be used only with Python 3.8 and
higher**.

Yout can check out the version of your Python distribution, you can run the following
command in a PowerShell or CommandLine in Windows, or in a shell in Linux/MacOS:

.. code-block:: bash

    python --version

As any package, **PyRCN** has several dependencies as listed in the `requirements.txt
 <https://github.com/TUD-STKS/PyRCN/blob/main/requirements.txt>`_. To avoid any
unexpected interaction with the basic system as installed on your computer, we highly
recommend  using a virtual environment

You can find more information about virtual environments, by checking the `Python
documentation on virtual environments and packages
<https://docs.python.org/3/tutorial/venv.html>`_.

Installation using `pip`
------------------------

We uploaded PyRCN to the `PyPI index for Python packages <https://pypi.org/>`_. Thus,
you can simply download and install `PyRCN <https://pypi.org/project/pyrcn/>`_ using
the `pip` command in your terminal:

.. code-block:: bash

    pip install pyrcn

You can, of course, also upgrade an existing version of PyRCN using `pip` on your command
line:

.. code-block:: bash

    pip install --upgrade pyrcn


To check your installation of PyRCN, run:

.. code-block:: bash

    pip show pyrcn

In case of any problems, please report any bugs at the `issue tracker on GitHub`_ or just
send a mail to Peter Steiner `peter.steiner@pyrcn.net <mailto:peter.steiner@pyrcn.net>`_.

Installation from source
------------------------

We only recommend you the installation of PyRCN from source if you would like to
contribute to PyRCN. Therefore, please find the source code of **PyRCN** on `GitHub
<https://github.com/TUD-STKS/PyRCN>`_.

You can download the latest stable version from the ``main`` branch. To work with older
or unstable versions of **PyRCN**,. you can checkout the ``dev`` branch or any other
branch you would like **PyRCN** from.

For the actual installation, please unzip the downloaded file or clone the repository.
The installation then work similar as before using `pip` in your command line:

.. code-block:: bash

    pip install -e /path/to/pyrcn

.. _issue tracker on GitHub: https://github.com/TUD-STKS/PyRCN/issues
