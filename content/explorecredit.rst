pipe it up!
############

:date: 2016-04-15 5:08
:category: project
:tags: machine learning, code, notebook
:slug: pipeline
:authors: Rebeccah Duvoisin
:summary: Building a pipeline for machine learning projects
:headline: Building a pipeline for machine learning projects
:header_cover: images/pairplot.png



Elements of a machine learning pipeline 
================================================================================

- Acquire raw data (`read-in data <{filename}/oopml.rst#read>`_.)
- Explore
- Train |recycle|
  + |Process|_
  + Cross-validate models
-  Test


.. |recycle| image:: {filename}/images/cycle.png
    :scale: 60 %
    :height: 20px


The focus of this post are the **(pre-)processing** steps of the data pipeline.  As our muse, we are using data from an open-source ML project ("Give me some credit") to predict serious delinquency of credit borrowers. 

Exploratory plots like this one help us prioritize data munging efforts.


.. {% notebook notebooks/pa2.ipynb %}

.. See |Trainer| defintion.

.. |Trainer| replace:: ``Trainer``
.. _Trainer: {filename}/oopml.rst#trainer>

- |pairplot|_

.. - |precision|_
.. - |roc_by_classifier|_

.. |pairplot| replace:: **Pair plot**
.. _pairplot:

.. figure:: {filename}/images/pairplot.png
    :alt: pair_plot
    :align: center
    :scale: 60 %
    :height: 1000px

    Sea born pair plot of features by delinquency (green)


.. |Process| replace:: **Process data**
.. _Process:

Processing
--------------

- |drop|_
- |transform|_

.. |drop| replace:: **Missing Data**
.. _drop:


Handling Missing Values
************************

The following function is adapted for objects developed in an earlier post on `obeject-oriented programming <{filename}/oopml.rst>`_ for machine learning projects. The function generates several modified datasets from a single ``Trainer`` object (training set). These variations of the original training set are derived from either dropping rows or dropping features.  This particular function saves these alternative training sets as additional ``Trainer`` objects, and indexes them to a ``ModelTrains`` object (`Modeling <{filename}/oopml.rst#modeling>`_.  

.. code-include:: ../../cappml/pa3/mlpipeline_pa3.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 184
    :end-line: 252


.. |transform| replace:: **Feature transformation**
.. _transform:

Variable Transformation
-------------------------

``gen_transform_data`` generates a ``Trainer`` object with transformed variables, saving it to ``ModelTrains``.

.. code-include:: ../../cappml/pa3/mlpipeline_pa3.py
    :lexer: python
    :encoding: utf-8
    :tab-width: 4
    :start-line: 254
    :end-line: 296


`See earlier post on OOP tips for data preparation <{filename}/oopml.rst>`_ for machine learning projects. 
