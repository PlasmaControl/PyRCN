import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyRCN",
    version="0.0.4",
    author="Peter Steiner",
    author_email="peter.steiner@tu-dresden.de",
    description="A Python3 framework for Reservoir Computing with a scikit-learn-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TUD-STKS/PyRCN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    keywords='PyRCN, Echo State Network',
    project_urls={
        'Documentation': 'https://pyrcn.readthedocs.io/',
        'Funding': 'https://pyrcn.readthedocs.io/',
        'Source': 'https://github.com/TUD-STKS/PyRCN',
        'Tracker': 'https://github.com/TUD-STKS/PyRCN/issues',
    },
    install_requires=[
        'scikit-learn>=0.22.1',
        'numpy>=1.18.1',
        'scipy>=1.2.0',
        'joblib>=0.13.2',
    ],
    python_requires='>=3.6',
)
