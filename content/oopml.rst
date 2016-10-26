you down with oop?
##################

:date: 2016-04-01 5:08
:category: project
:tags: machine learning, code
:slug: oop-ml
:authors: Rebeccah Duvoisin
:summary: object-oriented programming for machine learning projects
:headline: you know me!
:header_cover: images/matrix_FULL_MISS_4_crop_dark.png


Classes can help to shape your machine learning procedure
================================================================================

- |Trainer|_
- |Modeling|_
- |Smart|_  

Machine Learning projects typically require iterating through numerous subsets of your data in order to cross-validate models on training and testing data.  Python's `Scikit-Learn <http://Scikit-learn.org/stable/>`_'s library allows users to push their data through a loop of out-of-the-box models for selection quite easily.  

	Documenting the creation and evaluation of these subsets within that loop can be dizzying, particularly where preprocessing and multiple imputation are necessary and depend on the completeness of the given subset.  

Therefore, while your results may (not) be clear *at the end*, dynamic documention of analyses on these interim subsets *during* the loop often requires heavy lifting in the modeling loop and overly complicates the code.  


Write objects to envelope your data (raw, meta, and learned)
================================================================================

The code below exemplifies a suite of classes to envelope data into robust objects, namely, |Trainer|_  and |Modeling|_  classes so that it can extracted at the appropriate points of the model selection loop.  

.. |Trainer| replace:: **Trainer**
.. _Trainer:

``Trainer`` Class
-------------------

Having a class dedicated to storing important metadata about your dataset can facilitate looping through models.  This ``Trainer`` object is a wrapper for a Pandas Dataframe.  

It holds attributes about a testing, training or holdout set.  A ``Trainer`` object not only holds the dataset in a dataframe, it also holds the dataset's metadata, telling us which varaible is the outcome variable to classify, and which variables require imputation.  This enables us to dynamically query the metadata to determine which preprocessing steps must be taken to prepare a given dataset for modeling:

.. code-include:: ../../cappml/pa3/model.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 248
    :end-line: 599


.. |Modeling| replace:: **ModelTrains**
.. _Modeling:

``Modeling`` Class
-------------------

A modeling class, ``ModelTrains``, catalogues all variants of training, testing and validation ``Trainer`` objects.  Many subsets are created through the iterative cross-validation steps of a loop of models.  ``ModelTrains`` maintains and updates the geneology of these subsets.  Using this portfolio of subsets, we can (re)assign subsets as testing sets of one another dynamically simply by manipulating their relational attributes:

.. code-include:: ../../cappml/pa3/model.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 602
    :end-line: 836


.. |Smart| replace:: **Smarter functions**
.. _Smart:

Adapt functions for Trainer and ModelTrains objects
==================================================================

.. - |Read|_
.. - |drop|_
.. - |transform|_

.. |Read| replace:: **Read**
.. _Read:

Read-in Function
------------------

The ``Trainer`` class enables us a generalizeable function for reading in raw data directly into testing, training and holdout data sets - all as ``Trainer`` objects.  ``Trainer`` ojects record whether their data is a subsets of another ``Trainer`` object so that heritage can be easily retraced:

.. code-include:: ../../cappml/pa3/model.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 210
    :end-line: 246

.. .. |drop| replace:: **Remove Missings**
.. .. _drop:

.. Handling Missing Values
.. -------------------------

.. The following function produces several variants of a single ``Trainer`` object as additional ``Trainer``'s, and saves them to ``ModelTrains`` by name.  

.. .. code-include:: ../../cappml/pa3/mlpipeline_pa3.py
..     :lexer: python
..     :encoding: utf-8
..     :tab-width: 4
..     :start-line: 184
..     :end-line: 252


.. .. |transform| replace:: **Transform**
.. .. _transform:

.. Variable Transformation
.. -------------------------

.. ``gen_transform_data`` generates a ``Trainer`` object with transformed variables, saving it to ``ModelTrains``.

.. .. code-include:: ../../cappml/pa3/mlpipeline_pa3.py
..     :lexer: python
..     :encoding: utf-8
..     :tab-width: 4
..     :start-line: 254
..     :end-line: 296
