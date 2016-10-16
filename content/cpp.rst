python for data management 
##################################

:date: 2016-01-16 5:09
:tags: public policy, code
:category: project
:slug: crime
:authors: Rebeccah Duvoisin
:summary: 2015 Chicago Crime Data continued
:header_cover: images/elsalvador.jpg
:headline: Code for 2015 Chicago Crime Data


We look at 2015 crime data across Chicago's community areas using structured csv's. 

`See earlier notebook on pandas for csv's <{filename}/inspect.md>`_


Begin with a CommunityArea class to store and display analytic attributes:

.. code-include:: ../../cpp/chi_crime/elaborate.py
	:lexer: python
	:encoding: utf-8
	:tab-width: 4
	:start-line: 1
	:end-line: 33


Add a CityData object to read data:

.. code-include:: ../../cpp/chi_crime/elaborate.py
	:lexer: python
	:encoding: utf-8
	:tab-width: 4
	:start-line: 142
	:end-line: 161

Give it a few helpful behaviors and add more if you like:

.. code-include:: ../../cpp/chi_crime/elaborate.py
	:lexer: python
	:encoding: utf-8
	:tab-width: 4
	:start-line: 243
	:end-line: 312

Lastly, build your standard Coordinate class to handle lat/lon locating fields:

.. code-include:: ../../cpp/chi_crime/elaborate.py
	:lexer: python
	:encoding: utf-8
	:tab-width: 4
	:start-line: 313
	:end-line: 363
