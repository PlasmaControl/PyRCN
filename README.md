# PyRCN
A Python 3 framework for Reservoir Computing with a scikit-learn-compatible API.

PyRCN is a light-weight and transparent Python 3 framework that implements ESNs and is based on widely used scientific Python packages, such as numpy or scipy. The API is fully scikit-learn-compatible, so that users of scikit-learn do not need to restructure their research data in order to use ESNs. Interested used can directly use scikit-learns built-in parameter optimization methods and example datasets.

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

# Installation

## Prerequisites

PyRCn is developed using Python 3.6 or newer. It depends on the following packages:

- [numpy>=1.18.1](https://numpy.org/)
- [scipy>=1.2.0](https://scipy.org/)
- [scikit-learn>=0.22.1](https://scikit-learn.org/stable/)
- [joblib>=0.13.2](https://joblib.readthedocs.io)

## Installation from PyPI

The easiest and recommended way to install ``PyRCN`` is to use ``pip`` from (PyPI)[https://pypi.org] :
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

- ´doc´
    - This folder includes the package documentation.
- ´examples´
    - This folder includes example code as Jupyter Notebooks and python scripts.
- ´images´
    - This folder includes the logos used in ´README.md´.
- ´pyrcn´
    - This folder includes the actual Python package.


# Getting Started

PyRCN includes currently two variantes of Echo State Networks (ESNs): The ESNClassifier is meant to be a classifier, the ESNRegressor is meant to be a regressor.

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

# Citation

If you use PyRCN, please cite the following publication:

```latex
@INPROCEEDINGS{src:Steiner-20c,  
    author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone Peter Birkholz},  
    booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
    title={PyRCN: Exploration and Application of ESNs},  
    year={2020},  
    note={submitted},
}
```

# References:

PyRCN: Exploration and Application of ESNs

```latex
@INPROCEEDINGS{src:Steiner-20c,  
    author={Peter Steiner and Azarakhsh Jalalvand and Simon Stone Peter Birkholz},  
    booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
    title={PyRCN: Exploration and Application of ESNs},  
    year={2020},  
    note={submitted},
}
```

Note Onset Detection using Echo State Networks

```latex
@InProceedings{src:Steiner-20a,
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

Feature Engineering and Stacked ESNs for Musical Onset Detection

```latex
@INPROCEEDINGS{src:Steiner-20d,  
    author={Peter Steiner and Simon Stone and Azarakhsh Jalalvand and Peter Birkholz},  
    booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
    title={Feature Engineering and Stacked ESNs for Musical Onset Detection},  
    year={2020},  
    volume={},  
    number={},  
    note={submitted},
}
```

Multipitch tracking in music signals using Echo State Networks
```latex
@INPROCEEDINGS{src:Steiner-20b,
    author={Peter Steiner and Simon Stone and Peter Birkholz and Azarakhsh Jalalvand},
    booktitle={28th European Signal Processing Conference (EUSIPCO), 2020},
    title={Multipitch tracking in music signals using Echo State Networks},
    year={2020},
    note={accepted},
}
```

Multiple-F0 {E}stimation using {E}cho {S}tate {N}etworks
```latex
@inproceedings{src:Steiner-19,
  title={Multiple-F0 {E}stimation using {E}cho {S}tate {N}etworks},
  booktitle={{MIREX}},
  author={Peter Steiner and Azarakhsh Jalalvand and Peter Birkholz},
  year={2019},
  url = {https://www.music-ir.org/mirex/abstracts/2019/SBJ1.pdf}
}
```


# Acknowledgements
This research is financed by Europäischer Sozialfonds (ESF), the Free State of Saxony and Ghent University.

![SMWA_EFRE-ESF Logo](https://github.com/TUD-STKS/PyRCN/blob/master/images/SMWA_EFRE-ESF_Sachsen_Logokombi_quer_03.jpg)
[comment]: <> (This is a comment, it will not be included ![Ghent University](https://github.com/TUD-STKS/PyRCN/blob/master/images/logo_UGent_EN_RGB_2400_color-on-white.png) ![IDLab](https://github.com/TUD-STKS/PyRCN/blob/master/images/Logo_IDLab_White.png)) 
