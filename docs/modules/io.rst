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

.. autofunction:: trackintel.io.file.read_tours_csv

GeoDataFrame Import
=============================

.. autofunction:: trackintel.io.from_geopandas.read_positionfixes_gpd

.. autofunction:: trackintel.io.from_geopandas.read_triplegs_gpd

.. autofunction:: trackintel.io.from_geopandas.read_staypoints_gpd

.. autofunction:: trackintel.io.from_geopandas.read_locations_gpd

.. autofunction:: trackintel.io.from_geopandas.read_trips_gpd

.. autofunction:: trackintel.io.from_geopandas.read_tours_gpd


PostGIS Import
==============

.. autofunction:: trackintel.io.postgis.read_positionfixes_postgis

.. autofunction:: trackintel.io.postgis.read_triplegs_postgis

.. autofunction:: trackintel.io.postgis.read_staypoints_postgis

.. autofunction:: trackintel.io.postgis.read_locations_postgis

.. autofunction:: trackintel.io.postgis.read_trips_postgis

.. autofunction:: trackintel.io.postgis.read_tours_postgis

CSV File Export
===============

.. autofunction:: trackintel.io.file.write_positionfixes_csv

.. autofunction:: trackintel.io.file.write_triplegs_csv

.. autofunction:: trackintel.io.file.write_staypoints_csv

.. autofunction:: trackintel.io.file.write_locations_csv

.. autofunction:: trackintel.io.file.write_trips_csv

.. autofunction:: trackintel.io.file.write_tours_csv

PostGIS Export
==============

.. autofunction:: trackintel.io.postgis.write_positionfixes_postgis

.. autofunction:: trackintel.io.postgis.write_triplegs_postgis

.. autofunction:: trackintel.io.postgis.write_staypoints_postgis

.. autofunction:: trackintel.io.postgis.write_locations_postgis

.. autofunction:: trackintel.io.postgis.write_trips_postgis

.. autofunction:: trackintel.io.postgis.write_tours_postgis

Predefined dataset readers
==========================
We also provide functionality to parse well-known datasets directly into the trackintel framework.

Geolife
-----------
We support easy parsing of the Geolife dataset including available mode labels.

.. autofunction:: trackintel.io.dataset_reader.read_geolife

.. autofunction:: trackintel.io.dataset_reader.geolife_add_modes_to_triplegs
