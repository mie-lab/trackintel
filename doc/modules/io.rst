Input/Output
************

We primarily support two types of data ingestion:

* From CSV files.
* From PostGIS databases.

CSV File Import
===============

.. autofunction:: trackintel.io.file.read_positionfixes_csv

PostGIS Import
==============

.. autofunction:: trackintel.io.postgis.read_positionfixes_postgis

.. autofunction:: trackintel.io.postgis.write_positionfixes_postgis