Input/Output
************

We primarily support three types of data persistence:

* From CSV files.
* From `GeoDataFrames <https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.html#geopandas.GeoDataFrame>`_
* From PostGIS databases.

Our primary focus lies on supporting PostGIS databases for persistence, but of course you 
can use the standard Pandas/Python tools to persist your data to any database with a 
minimal bit of tweaking. And of course you can also keep all data in memory while you do 
an analysis, e.g., in a Jupyter notebook.

All the read/write functions are made available in the top-level ``trackintel`` module, i.e.,
you can use them as ``trackintel.read_positionfixes_csv('data.csv')``, etc. Note that these
functions are wrappers around the (Geo)Pandas CSV, renaming and SQL functions. As such, all ``*args``
and ``**kwargs`` are forwarded to them.

CSV File Import
===============

.. autofunction:: trackintel.io.file.read_positionfixes_csv

.. autofunction:: trackintel.io.file.read_triplegs_csv

.. autofunction:: trackintel.io.file.read_staypoints_csv

.. autofunction:: trackintel.io.file.read_locations_csv

.. autofunction:: trackintel.io.file.read_trips_csv

GeoDataFrame Import
=============================

.. autofunction:: trackintel.io.from_geopandas.positionfixes_from_gpd

.. autofunction:: trackintel.io.from_geopandas.triplegs_from_gpd

.. autofunction:: trackintel.io.from_geopandas.staypoints_from_gpd

.. autofunction:: trackintel.io.from_geopandas.locations_from_gpd

.. autofunction:: trackintel.io.from_geopandas.trips_from_gpd


PostGIS Import
==============

.. autofunction:: trackintel.io.postgis.read_positionfixes_postgis

.. autofunction:: trackintel.io.postgis.read_triplegs_postgis

.. autofunction:: trackintel.io.postgis.read_staypoints_postgis

.. autofunction:: trackintel.io.postgis.read_locations_postgis

.. autofunction:: trackintel.io.postgis.read_trips_postgis

CSV File Export
===============

.. autofunction:: trackintel.io.file.write_positionfixes_csv

.. autofunction:: trackintel.io.file.write_triplegs_csv

.. autofunction:: trackintel.io.file.write_staypoints_csv

.. autofunction:: trackintel.io.file.write_locations_csv

.. autofunction:: trackintel.io.file.write_trips_csv

PostGIS Export
==============

.. autofunction:: trackintel.io.postgis.write_positionfixes_postgis

.. autofunction:: trackintel.io.postgis.write_triplegs_postgis

.. autofunction:: trackintel.io.postgis.write_staypoints_postgis

.. autofunction:: trackintel.io.postgis.write_locations_postgis

.. autofunction:: trackintel.io.postgis.write_trips_postgis
