===========
Development
===========

PyRCN is an open-source project, and we highly welcome any contribution! There
are many ways to help, from reporting a problem to contributing code.

For general feedback, questions or ideas, you can always send a mail to Peter
Steiner (`peter.steiner@pyrcn.net <mailto:peter.steiner@pyrcn.net>`_). Most of
the development happens on `GitHub <https://github.com/PlasmaControl/PyRCN>`_.

How can you contribute?
=======================

Give feedback
-------------

For general feedback, questions or ideas for improvement, send a mail to Peter
Steiner (`peter.steiner@pyrcn.net <mailto:peter.steiner@pyrcn.net>`_) or open a
topic on the `issue tracker on GitHub`_.

Report bugs
-----------

Please report any bugs at the `issue tracker on GitHub`_. A good bug report
includes the PyRCN and Python versions, a short description of the expected and
the observed behaviour, and, if possible, a minimal example that reproduces the
problem.

If you are unsure whether the behaviour you experienced is intended or a bug,
feel free to send a mail to Peter Steiner
(`peter.steiner@pyrcn.net <mailto:peter.steiner@pyrcn.net>`_) first.

Improve the documentation
--------------------------

Whenever you find something that is not explained well, is out of date or is
simply missing, please let us know through the `issue tracker on GitHub`_ or by
mail. Documentation fixes are contributed as code (see below), and even small
corrections are very welcome.

Contributing code
==================

Code contributions are made through pull requests on GitHub. The development
branch is ``dev``; the ``main`` branch always holds the latest released version.

Setting up a development environment
------------------------------------

Fork the repository on GitHub, then clone your fork and install PyRCN in
editable mode together with the ``test`` extra, ideally in a fresh virtual
environment:

.. code-block:: bash

    git clone https://github.com/<your-username>/PyRCN.git
    cd PyRCN
    python -m venv .venv
    source .venv/bin/activate  # on Windows: .venv\Scripts\activate
    pip install -e .[test]

The ``test`` extra installs the tools used by the continuous integration:
pytest, pytest-cov, mypy and flake8.

Running the checks
------------------

Before opening a pull request, please make sure the same checks that run in
continuous integration pass locally. Lint the code, type-check it and run the
test suite:

.. code-block:: bash

    flake8 src/pyrcn tests
    mypy src/pyrcn
    pytest

Contribution workflow
---------------------

1. Fork the repository and create a feature branch off ``dev``:

   .. code-block:: bash

       git switch dev
       git switch -c my-feature

2. Make your changes and add tests that cover them.
3. Run the checks above (flake8, mypy and pytest) until they all pass.
4. Commit your work and push the branch to your fork:

   .. code-block:: bash

       git push -u origin my-feature

5. Open a pull request against the ``dev`` branch of the main repository and
   describe what your change does and why.

A maintainer will review the pull request and may ask for adjustments before it
is merged.

.. _GitHub: https://github.com/PlasmaControl/PyRCN
.. _issue tracker on GitHub: https://github.com/PlasmaControl/PyRCN/issues
