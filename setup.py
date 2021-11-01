import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyRCN",
    version="0.0.14",
    author="Peter Steiner",
    author_email="peter.steiner@tu-dresden.de",
    description="A scikit-learn-compatible framework for Reservoir Computing in Python",
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
        'scikit-learn>=0.23.1',
        'ipywidgets',
        'ipympl',
        'numpy>=1.18.1',
        'scipy>=1.2.0',
        'joblib>=0.13.2',
        'pandas>=1.0.0',
        'matplotlib',
        'seaborn',
        'tqdm>=4.33.0',
    ],
    python_requires='>=3.7',
)
