The *trackintel* Documentation
******************************

.. image:: https://img.shields.io/badge/version-v0.2.0-red.svg

Focusing on human mobility data, *trackintel* provides functionalities for data quality 
enhancement, integrating data from various sources, performing quantitative analysis and 
mining tasks, and visualizing the data and/or analysis results. In addition to these 
core functionalities, packages are provided for user mobility profiling and trajectory-based 
learning analytics. It is split into the different steps of a typical processing pipeline, 
and assumes that data is available adhering to the trackintel data model format:

* Preprocessing (filtering, outlier detection, imputation of missing values)
* Contextual Augmentation (map matching, trajectory algebra-based context addition)
* Analysis (extraction of mobility metrics and descriptors, preferences, systematic mobility)
* Visualization and Communication (generation of maps, charts, etc.)
* Non-standardized methods and algorithms are explicitly denoted as experimental and 
  (whenever possible) separated from the standardized methods.

For information about the trackintel data models that are used throughout the framework, 
please refer to the :doc:`/modules/model` page. For a quick deployment to a PostGIS database, you can 
use the SQL commands given at :doc:`/content/data_model_sql` or run the file found 
`on Github <https://github.com/mie-lab/trackintel/blob/master/sql/create_tables_pg.sql>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorial
   modules/model
   modules/io
   modules/preprocessing
   modules/analysis
   modules/visualization
   modules/geogr



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
