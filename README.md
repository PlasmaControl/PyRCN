# PyRCN
A Python 3 framework for Reservoir Computing with a [scikit-learn](https://scikit-learn.org/stable/)-compatible API.

PyRCN ("Python Reservoir Computing Networks") is a light-weight and transparent Python 3 framework for Reservoir Computing (currently only implementing Echo State Networks) and is based on widely used scientific Python packages, such as numpy or scipy. The API is fully scikit-learn-compatible, so that users of scikit-learn do not need to refactor their code in order to use the estimators implemented by this framework. Scikit-learn's built-in parameter optimization methods and example datasets can also be used in the usual way.

PyRCN is used by the [Chair of Speech Technology and Cognitive Systems, Institute for Acoustics and Speech Communications, Technische Universität Dresden, Dresden, Germany](https://tu-dresden.de/ing/elektrotechnik/ias/stks?set_language=en) and [IDLab (Internet and Data Lab), Ghent University, Ghent, Belgium](https://www.ugent.be/ea/idlab/en). 

Currently, it implements Echo State Networks (ESNs) by Herbert Jaeger in different flavors, e.g. Classifier and Regressor. It is actively developed to be extended into several directions:

- Incorporate Feedback
- Better sequence handling with [sktime](http://sktime.org/)
- A unified API to stack ESNs
- More towards future work: Related architectures, such as Liquid State Machines (LSMs) and Perturbative Neural Networks (PNNs)

PyRCN has successfully been used for several tasks:

- Music Information Retrieval (MIR)
    - Multipitch tracking
    - Onset detection
- Time Series Prediction
    - Mackey-Glass benchmark test
    - Stock price prediction
- Ongoing research tasks:
    - Beat tracking in music signals
    - Pattern recognition in sensor data
    - Phoneme recognition

Please see the [References section](#references) for more information. For code examples, see [Getting started](#getting-started).

# Installation

## Prerequisites

PyRCn is developed using Python 3.6 or newer. It depends on the following packages:

- [numpy>=1.18.1](https://numpy.org/)
- [scipy>=1.2.0](https://scipy.org/)
- [scikit-learn>=0.23.1](https://scikit-learn.org/stable/)
- [joblib>=0.13.2](https://joblib.readthedocs.io)
- [pandas>=1.0.0](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
## Installation from PyPI

The easiest and recommended way to install ``PyRCN`` is to use ``pip`` from [PyPI](https://pypi.org) :

```python
pip install pyrcn   
```

## Installation from source

If you plan to contribute to ``PyRCN``, you can also install the package from source.

Clone the Git repository:

```
git clone https://github.com/TUD-STKS/PyRCN.git
```

Install the package using ``setup.py``:
```
python setup.py install --user
```

# Package structure
The package is structured in the following way: 

- `doc`
    - This folder includes the package documentation.
- `examples`
    - This folder includes example code as Jupyter Notebooks and python scripts.
- `images`
    - This folder includes the logos used in ´README.md´.
- `pyrcn`
    - This folder includes the actual Python package.


# Getting Started

PyRCN includes currently variants of Echo State Networks (ESNs) and Extreme Learning Machines (ELMs): Regressors and Classifiers.

Basic example for the ESNClassifier:

```python
from pyrcn.echo_state_network import ESNClassifier


clf = ESNClassifier()
clf.fit(X=X_train, y=y_train)

y_pred_classes = clf.predict(X=X_test)  # output is the class for each input example
y_pred_proba = clf.predict_proba(X=X_test)  #  output are the class probabilities for each input example
```

Basic example for the ESNRegressor:

```python
from pyrcn.echo_state_network import ESNRegressor


reg = ESNRegressor()
ref.fit(X=X_train, y=y_train)

y_pred = reg.predict(X=X_test)  # output is the prediction for each input example
```

An extensive introduction to getting started with PyRCN is included in the [examples](https://github.com/TUD-STKS/PyRCN/blob/master/examples) directory. The notebook [digits](https://github.com/TUD-STKS/PyRCN/blob/master/examples/digits.ipynb) or its corresponding [Python script](https://github.com/TUD-STKS/PyRCN/blob/master/examples/digits.py) show how to set up an ESN for a small hand-written digit recognition experiment.

Launch the digits notebook on Binder: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TUD-STKS/PyRCN/master?filepath=examples%2Fdigits.ipynb)

Fore more advanced examples, please have a look at our [Automatic Music Transcription Repository](https://github.com/TUD-STKS/Automatic-Music-Transcription), in which we provide an entire feature extraction, training and test pipeline for multipitch tracking and for note onset detection using PyRCN.

# Citation

If you use PyRCN, please cite the following publication:

```latex
@misc{steiner2021pyrcn,
      title={PyRCN: Exploration and Application of ESNs}, 
      author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone and Peter Birkholz},
      year={2021},
      eprint={2103.04807},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# References

[Unsupervised Pretraining of Echo State Networks for Onset Detection](https://link.springer.com/chapter/10.1007/978-3-030-85099-9_12)

```
@InProceedings{src:Steiner-21e,
   author="Peter Steiner and Azarakhsh Jalalvand and Peter Birkholz",
   editor="Igor Farka{\v{s}} and Paolo Masulli and Sebastian Otte and Stefan Wermter",
   title="{U}nsupervised {P}retraining of {E}cho {S}tate {N}etworks for {O}nset {D}etection",
   booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2021",
   year="2021",
   publisher="Springer International Publishing",
   address="Cham",
   pages="59--70",
   isbn="978-3-030-86383-8"
}


```

[Improved Acoustic Modeling for Automatic Piano Music Transcription Using Echo State Networks](https://link.springer.com/chapter/10.1007/978-3-030-85099-9_12)

```
@InProceedings{src:Steiner-21d,
	author="Peter Steiner and Azarakhsh Jalalvand and Peter Birkholz",
	editor="Ignacio Rojas and Gonzalo Joya and Andreu Catala",
	title="{I}mproved {A}coustic {M}odeling for {A}utomatic {P}iano {M}usic {T}ranscription {U}sing {E}cho {S}tate {N}etworks",
	booktitle="Advances in Computational Intelligence",
	year="2021",
	publisher="Springer International Publishing",
	address="Cham",
	pages="143--154",
	isbn="978-3-030-85099-9"
}
```

[Glottal Closure Instant Detection using Echo State Networks](http://www.essv.de/pdf/2021_161_168.pdf)

```latex
@InProceedings{src:Steiner-21c,
	title = {Glottal Closure Instant Detection using Echo State Networks},
	author = {Peter Steiner and Ian S. Howard and Peter Birkholz},
	year = {2021},
	pages = {161--168},
	keywords = {Oral},
	booktitle = {Studientexte zur Sprachkommunikation: Elektronische Sprachsignalverarbeitung 2021},
	editor = {Stefan Hillmann and Benjamin Weiss and Thilo Michael and Sebastian Möller},
	publisher = {TUDpress, Dresden},
	isbn = {978-3-95908-227-3}
} 
```

Cluster-based Input Weight Initialization for Echo State Networks

```latex
@misc{src:Steiner-KM_ESN,
    title={Cluster-based Input Weight Initialization for Echo State Networks},
    author={Peter Steiner and Azarakhsh Jalalvand and Peter Birkholz},
    year={2021},
    eprint={2103.04710},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

PyRCN: Exploration and Application of ESNs

```latex
@misc{src:Steiner-PyRCN,
      title={PyRCN: Exploration and Application of ESNs}, 
      author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone and Peter Birkholz},
      year={2021},
      eprint={2103.04807},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[Feature Engineering and Stacked ESNs for Musical Onset Detection](https://ieeexplore.ieee.org/abstract/document/9413205)

```latex
@INPROCEEDINGS{src:Steiner-21b,
    author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone and Peter Birkholz},
    booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
    title={{F}eature {E}ngineering and {S}tacked {E}cho {S}tate {N}etworks for {M}usical {O}nset {D}etection},
    year={2021},
    volume={},
    number={},
    pages={9537--9544},
    doi={10.1109/ICPR48806.2021.9413205}
}
```

[Multipitch tracking in music signals using Echo State Networks](https://ieeexplore.ieee.org/abstract/document/9287638)

```latex
@INPROCEEDINGS{src:Steiner-21a,
    author={Peter Steiner and Simon Stone and Peter Birkholz and Azarakhsh Jalalvand},
    booktitle={2020 28th European Signal Processing Conference (EUSIPCO)},
    title={{M}ultipitch tracking in music signals using {E}cho {S}tate {N}etworks},
    year={2021},
    volume={},
    number={},
    pages={126--130},
    keywords={},
    doi={10.23919/Eusipco47968.2020.9287638},
    ISSN={2076-1465},
    month={Jan},
```

[Note Onset Detection using Echo State Networks](http://www.essv.de/pdf/2020_157_164.pdf)

```latex
@InProceedings{src:Steiner-20,
	title = {Note Onset Detection using Echo State Networks},
	author = {Peter Steiner and Simon Stone and Peter Birkholz},
	year = {2020},
	pages = {157--164},
	keywords = {Poster},
	booktitle = {Studientexte zur Sprachkommunikation: Elektronische Sprachsignalverarbeitung 2020},
	editor = {Ronald Böck and Ingo Siegert and Andreas Wendemuth},
	publisher = {TUDpress, Dresden},
	isbn = {978-3-959081-93-1}
} 
```

[Multiple-F0 Estimation using Echo State Networks](https://www.music-ir.org/mirex/abstracts/2019/SBJ1.pdf)

```latex
@inproceedings{src:Steiner-19,
  title={{M}ultiple-{F}0 {E}stimation using {E}cho {S}tate {N}etworks},
  booktitle={{MIREX}},
  author={Peter Steiner and Azarakhsh Jalalvand and Peter Birkholz},
  year={2019},
  url = {https://www.music-ir.org/mirex/abstracts/2019/SBJ1.pdf}
}
```


# Acknowledgements
This research was funded by the European Social Fund (Application number: 100327771) and co-financed by tax funds based on the budget approved by the members of the Saxon State Parliament, and by Ghent University.
