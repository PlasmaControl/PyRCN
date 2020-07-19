Usage
=====

Library usage
-------------

To use PyRCN, we strongly recommend :ref:`installing it from source <install_from_source>`. Installation from package works as well, but you might end up using an older version of PyRCN.

PyRCN includes currently two variants of Echo State Networks (ESNs): The ESNClassifier is meant to be a classifier, the ESNRegressor is meant to be a regressor.

Basic example for the ESNClassifier::

  from pyrcn.echo_state_network import ESNClassifier
  
  
  clf = ESNClassifier()
  clf.fit(X=X_train, y=y_train)
  
  y_pred_classes = clf.predict(X=X_test)  # output is the class for each input example
  y_pred_proba = clf.predict_proba(X=X_test)  #  output are the class probabilities for each input example


Basic example for the ESNRegressor::

  from pyrcn.echo_state_network import ESNRegressor
  
  
  clf = ESNClassifier()
  clf.fit(X=X_train, y=y_train)
  
  y_pred = clf.predict(X=X_test)  # output is the prediction for each input example


To learn more about how to use the library please follow the
:doc:`tutorials <tutorial>`.
