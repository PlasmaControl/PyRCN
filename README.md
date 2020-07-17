# PyRCN
A Python 3 framework for Reservoir Computing with a scikit-learn-compatible API.

PyRCN is a light-weight and transparent Python 3 framework that implements ESNs and is based on widely used scientific Python packages, such as numpy or scipy. The API is fully scikit-learn-compatible, so that users of scikit-learn do not need to restructure their research data in order to use ESNs. Interested used can directly use scikit-learns built-in parameter optimization methods and example datasets.

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

y_pred_classes = reg.predict(X=X_test)  # output is the prediction for each input example
```

# Acknowledgements
This research is financed by Europäischer Sozialfonds (ESF), the Free State of Saxony and Ghent University.

![SMWA_EFRE-ESF Logo](https://github.com/TUD-STKS/PyRCN/blob/master/images/SMWA_EFRE-ESF_Sachsen_Logokombi_quer_03.jpg)

# References
If you use the PyRCN, please cite the following publication:

```latex
@INPROCEEDINGS{src:PyRCN-20,  
	author={Peter Steiner and Simon Stone and Azarakhsh Jalalvand and Peter Birkholz},  
	booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
	title={Feature Engineering and Stacked ESNs for Musical Onset Detection},  
	year={2020},  
	volume={},  
	number={},  
	note={submitted},
}
```
