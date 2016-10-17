give me some credit
####################

:date: 2016-04-30 5:08
:category: project
:tags: machine learning, code
:slug: Scikit-ml
:authors: Rebeccah Duvoisin
:summary: Classification with Scikit-Learn
:headline: Predicting delinquency on loans repayment
:header_cover: images/ROC_by_classifier_dark.png


`See post on OOP tips for data preparation <{filename}/oopml.rst>`_


This open-source ML project is to predict serious delinquency of credit borrowers.

One strategy is to build a master loop through scikit's library to learn the "best" model. For new learners, this angle some serious pedagogical drawbacks, but it is easily implemented. Here's the gist of the model selection loop to get you familiar with the objective here.  I've called it the ``splitter`` function because it splits the training set into cross-validation sets with each iteration of the model.  

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 221
    :end-line: 259

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 479
    :end-line: 511

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 540
    :end-line: 560

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 570
    :end-line: 573

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 584
    :end-line: 611


At each improvement of the evaluation metric, a new "best" model is saved. See |bestmodel|_ function.

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 612
    :end-line: 632

With each new "best" model, we take several snapshots of the model's performance, using the |bestmodel|_ function.

- |feature_importance|_
- |precision|_
- |roc_by_classifier|_

.. |feature_importance| replace:: **Feature Impotance**
.. _feature_importance:

.. figure:: {filename}/images/HOLDOUT_MISS_log_Final_Validation_RandomForestClassifier_feat_importance.png
    :alt: feat_importance
    :align: right
    :scale: 60 %
    :height: 1000px

    Feature importance of a RandomForest Classifier model.


.. `precision`_
.. |precision| replace:: **Precision-Recall**
.. _precision:

.. figure:: {filename}/images/HOLDOUT_MISS_log_Final_Validation_RandomForestClassifier_precision_recall_at_5.png
    :alt: precision_recall
    :align: right
    :scale: 40 %
    :height: 1000px

    Precision and recall curves by population percentage.


Lastly, |bestmodel|_ also stores the best learned model for each classifier tested so that we can compare their relative performances.


.. `roc_by_classifier`_
.. |roc_by_classifier| replace:: **ROC Curves**
.. _roc_by_classifier:

.. figure:: {filename}/images/ROC_by_classifier.png
    :alt: roc_curves
    :align: right
    :scale: 80 %
    :height: 1000px

    ROC curves by classifier.

.. |bestmodel| replace:: ``replace_best_model``
.. _bestmodel:

Selected Model(s) 
------------------

.. code-include:: ../../cappml/pa3/rayidsplitter.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 161
    :end-line: 221