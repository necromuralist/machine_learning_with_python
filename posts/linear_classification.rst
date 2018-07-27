.. title: Linear Classification
.. slug: Linear-Classification
.. date: 2017-07-13 12:38
.. tags: classification
.. link: 
.. description: Introduction to Linear Classifiers
.. type: text

1 Imports
---------

.. code:: ipython

    import pandas
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

2 The Data
----------

.. code:: ipython

    cancer = load_breast_cancer()
    print(cancer.keys())

::

    dict_keys(['target_names', 'feature_names', 'data', 'DESCR', 'target'])

.. code:: ipython

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                        cancer.target,
                                                        stratify=cancer.target)

3 Logistic Regression
---------------------

.. code:: ipython

    logistic_model = LogisticRegression(penalty="l1")
    logistic_model.fit(X_train, y_train)
    print("Logistic Training Accuracy: {:.2f}".format(logistic_model.score(X_train, y_train)))
    print("Logistic Testing Accuracy: {:.2f}".format(logistic_model.score(X_test, y_test)))

::

    Logistic Training Accuracy: 0.97
    Logistic Testing Accuracy: 0.92

Depending on the random seed it sometimes does better on the testing than it does on the training set.

.. code:: ipython

    coefficients = pandas.Series(logistic_model.coef_[0], index=cancer.feature_names)
    print(coefficients)

::

    mean radius                2.257338
    mean texture               0.058581
    mean perimeter            -0.001644
    mean area                 -0.009889
    mean smoothness            0.000000
    mean compactness           0.000000
    mean concavity             0.000000
    mean concave points        0.000000
    mean symmetry              0.000000
    mean fractal dimension     0.000000
    radius error               0.000000
    texture error              2.657975
    perimeter error            0.000000
    area error                -0.118846
    smoothness error           0.000000
    compactness error          0.000000
    concavity error            0.000000
    concave points error       0.000000
    symmetry error             0.000000
    fractal dimension error    0.000000
    worst radius               1.635063
    worst texture             -0.412327
    worst perimeter           -0.201013
    worst area                -0.022727
    worst smoothness           0.000000
    worst compactness          0.000000
    worst concavity           -4.246229
    worst concave points       0.000000
    worst symmetry             0.000000
    worst fractal dimension    0.000000
    dtype: float64

.. code:: ipython

    features = len(cancer.feature_names)
    non_zero = coefficients[coefficients!=0]
    print(non_zero)
    print(len(non_zero)/features)

::

    mean radius        2.257338
    mean texture       0.058581
    mean perimeter    -0.001644
    mean area         -0.009889
    texture error      2.657975
    area error        -0.118846
    worst radius       1.635063
    worst texture     -0.412327
    worst perimeter   -0.201013
    worst area        -0.022727
    worst concavity   -4.246229
    dtype: float64
    0.36666666666666664

The model was able to remove 37% of the features.

.. code:: ipython

    model = LogisticRegression(C=100)
    model.fit(X_train, y_train)
    print("Training Accuracy: {0:.2f}".format(model.score(X_train, y_train)))
    print("Testing Accuracy: {0:.2f}".format(model.score(X_test, y_test)))

::

    Training Accuracy: 0.98
    Testing Accuracy: 0.95

Using an *L2* penalty of 100 improves the accuracy of the model. Increasing "C" means less regularization, so in this case the improvement came from using a more complex model.

4 Support Vector Machine Classification
---------------------------------------

.. code:: ipython

    for power in range(-4, 4):
        penalty = 10**power
        svc = LinearSVC(C=penalty)
        svc.fit(X_train, y_train)
        print("C={}".format(penalty))
        print("Training Accuracy: {0:.2f}".format(svc.score(X_train, y_train)))
        print("Testing Accuracy: {0:.2f}".format(svc.score(X_test, y_test)))
        print()

::

    C=0.0001
    Training Accuracy: 0.93
    Testing Accuracy: 0.93

    C=0.001
    Training Accuracy: 0.93
    Testing Accuracy: 0.92

    C=0.01
    Training Accuracy: 0.70
    Testing Accuracy: 0.71

    C=0.1
    Training Accuracy: 0.94
    Testing Accuracy: 0.93

    C=1
    Training Accuracy: 0.92
    Testing Accuracy: 0.92

    C=10
    Training Accuracy: 0.93
    Testing Accuracy: 0.94

    C=100
    Training Accuracy: 0.86
    Testing Accuracy: 0.84

    C=1000
    Training Accuracy: 0.92
    Testing Accuracy: 0.92

Every time I run this it comes out slightly differently, but it seems like most values do pretty well, there's usually only one or two values of *C* below 0.92 for the test set.

5 Tuning the Penalty
--------------------

The L1 penalty makes use of more of the features so it will generally do better if they are all relevant. The L2 penalty is better for interpreting the important features and will do better if some of the features are in fact not relevant. Unlike *alpha* for regression, *C* decreases the regularization as it gets bigger. When searching for the best value it can be useful to search a logarithmic space (e.g. 0.001, 0.01, 0.1, 1, 10, 100)
