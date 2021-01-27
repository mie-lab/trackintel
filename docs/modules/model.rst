Model
*****

In trackintel, **tracking data** is split into several classes. It is not generally 
assumed that data is already available in all these classes, instead, trackintel
provides functionality to generate everything starting from the raw GPS positionfix data 
(consisting of at least ``(user_id, tracked_at, longitude, latitude)`` tuples).

* **users**: The users for which data is available.
* **positionfixes**: Raw GPS data.
* **staypoints**: Locations where a user spent a minimal time.
* **triplegs**: Segments covered with one mode of transport.
* **locations**: Clustered staypoints.
* **trips**: Segments between consecutive activity staypoints (special staypoints that are not just waiting points).
* **tours**: Sequences of trips which start and end at the same location (if ``journey`` 
  is set to ``True``, this location is *home*).

A detailed (and SQL-specific) explanation of the different classes can be found under 
:doc:`/content/data_model_sql`.

Some of the more time-consuming functions of trackintel generate logging data, as well as extracted 
features data, and they assume more data about geographic features or characteristics of transport 
modes are available. These are not explained here yet.

GeoPandas Implementation
========================

.. highlight:: python

In trackintel, we assume that all these classes are available as (Geo)Pandas (Geo)DataFrames. While we
do not extend the given DataFrame constructs, we provide accessors that validate that a given DataFrame
corresponds to a set of constraints, and make functions available on the DataFrames. For example::

    df = trackintel.read_positionfixes_csv('data.csv')
    df.as_positionfixes.extract_staypoints()

This will read a CSV into a format compatible with the trackintel understanding of a collection of 
positionfixes, and the second line will wrap the DataFrame with an accessor providing functions such 
as ``extract_staypoints()``. You can read up more on Pandas accessors in `the Pandas documentation 
<https://pandas.pydata.org/pandas-docs/stable/development/extending.html>`_.

Available Accessors
===================

The following accessors are available within *trackintel*.

.. autoclass:: trackintel.model.users.UsersAccessor

.. autoclass:: trackintel.model.positionfixes.PositionfixesAccessor

.. autoclass:: trackintel.model.staypoints.StaypointsAccessor

.. autoclass:: trackintel.model.triplegs.TriplegsAccessor

.. autoclass:: trackintel.model.locations.LocationsAccessor

.. autoclass:: trackintel.model.trips.TripsAccessor

.. autoclass:: trackintel.model.tours.ToursAccessor
