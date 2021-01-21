Preprocessing
*************

The preprocessing module contains a variety of functions to transform mobility and tracking 
data into richer data sources.

Filtering
=============
.. autofunction:: trackintel.preprocessing.filter.spatial_filter


Positionfixes
=============

As positionfixes are usually the data we receive from a tracking application of some sort,
there are various functions that extract meaningful information from it (and in the process
turn it into a higher-level *trackintel* data structure).

In particular, we can extract staypoints and triplegs from positionfixes.

.. autofunction:: trackintel.preprocessing.positionfixes.generate_staypoints

.. autofunction:: trackintel.preprocessing.positionfixes.generate_triplegs

Staypoints
==========

Staypoints are points where someone stayed for a longer period of time (e.g., during a
transfer between two transport modes). We can cluster these into locations that a user 
frequently visits.

.. autofunction:: trackintel.preprocessing.staypoints.generate_locations

Triplegs
========

Triplegs denote routes taken between two consecutive staypoint. Usually, these are traveled
with a single mode of transport. Depending on the tracking data, they can be rather noisy,
for which reason we often want to smoothen them.

.. autofunction:: trackintel.preprocessing.triplegs.smoothen_triplegs
.. autofunction:: trackintel.preprocessing.triplegs.generate_trips

The extraction of triplegs in `generate_trips` follows this algorithm:

.. image:: /_static/tripalgorithm.png
   :scale: 100 %
   :align: center
