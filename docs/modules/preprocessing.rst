Preprocessing
*************

The preprocessing module contains a variety of functions to transform mobility and tracking 
data into richer data sources.


Positionfixes
=============

As positionfixes are usually the data we receive from a tracking application of some sort,
there are various functions that extract meaningful information from it (and in the process
turn it into a higher-level *trackintel* data structure).

In particular, we can extract staypoints and triplegs from positionfixes.

.. autofunction:: trackintel.preprocessing.positionfixes.extract_staypoints

.. autofunction:: trackintel.preprocessing.positionfixes.extract_triplegs

Staypoints
==========

Staypoints are points where someone stayed for a longer period of time (e.g., during a
transfer between two transport modes). We can cluster these into locations that a user 
frequently visits.

.. autofunction:: trackintel.preprocessing.staypoints.cluster_staypoints

Triplegs
========

Triplegs denote routes taken between two consecutive staypoint. Usually, these are traveled
with a single mode of transport. Depending on the tracking data, they can be rather noisy,
for which reason we often want to smoothen them.

.. autofunction:: trackintel.preprocessing.triplegs.smoothen_triplegs
