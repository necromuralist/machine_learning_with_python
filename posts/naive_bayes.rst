.. title: Naive Bayes Classification
.. slug: Naive-Bayes-Classification
.. date: 2017-07-13 15:45
.. tags: classification
.. link: 
.. description: Naive Bayes example.
.. type: text
.. author: Brunhilde



1 Imports
---------

.. code:: ipython

    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

2 The Data
----------

.. code:: ipython

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                        cancer.target,
                                                        stratify=cancer.target)

3 The Model
-----------

.. code:: ipython

    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    print("Training Accuracy: {0:.2f}".format(bayes.score(X_train, y_train)))
    print("Testing Accuracy: {0:.2f}".format(bayes.score(X_test, y_test)))

::

    Training Accuracy: 0.95
    Testing Accuracy: 0.93

Naive Bayes works very fast and can handle very large sets of data, but it is called "naive" because it assumes that the features are all independent of each other and so it tends not to generalize as well as some other models. Since it's so efficient it can be used as a baseline to compare with other models.
