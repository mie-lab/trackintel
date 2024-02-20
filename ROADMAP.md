# Roadmap

This document explains the functionality planned for various releases.

## v2.0
* :heavy_check_mark: Parallelization.
* More method to predict activity without context data. [#213](https://github.com/mie-lab/trackintel/issues/213)
* Add context data to movement trajectories (using spatio-temporal map algebra). [#63](https://github.com/mie-lab/trackintel/issues/63) 
* :heavy_check_mark: Implement *tours* (starting and ending at a persons home location). [#287](https://github.com/mie-lab/trackintel/pull/287)
* :heavy_check_mark: Speed calculation for triplegs/positionfixes. [#191](https://github.com/mie-lab/trackintel/issues/191)
* CodePeerReview: Next milestone after 1.0. [#86](https://github.com/mie-lab/trackintel/issues/86) 
* Include the calculation of common mobility indicators (e.g., radius of gyration)

## Ideas that are on the list
I/O:
* :heavy_check_mark: Read *tours* from files.
* :heavy_check_mark: Read *tours* from PostGIS.
* :heavy_check_mark: Write *tours* to files.
* :heavy_check_mark: Write *tours* to PostGIS.
* Include data input/output methods for more common datasets

Preprocessing:
* Short walks to/from cars/buses/etc.: These are often not recognized by the tracking applications.
* Imputation of trivial gaps in tracking data (*triplegs* and *staypoints*).
* Simple outlier filtering for *triplegs*
* Create a solution to match trajectories from different tracking sources. [#49](https://github.com/mie-lab/trackintel/issues/49)

Augment tracking data:
* Better transport mode prediction based on movement data (speed, associated features from accelerometer data, etc.) as well as by aligning them with context data such as from OpenStreetMap or GTFS departure schedules. 
* Better activity prediction. This functionality is based on properties of the movement data and on context data such as points of interests.
* Provide holistic inference of transport modes (take into account the fact that if someone uses the car to reach a certain location, the person is likely to leave by car as well).
* Map match *triplegs* (based on transport mode identification). Solutions could be *LeuvenMapMatching* or based on *osrm*.

Analysis
* Mobility behaviour (and its changes): automatic detection of uncharacteristic changes in mobility patterns.
* User profiling and clustering.
* Anomaly detection (based on properties of movement data as well as contextual factors).
* Clustering of triplegs using similarity metrics
* Next place prediction (e.g., markov-model)

Visualization 
* Visualize *tours* geographically.
* Visualize *trips* and *locations* geographically, e.g., by coloring *triplegs* differently or by making a buffer around all *staypoints* that are part of a *locations*.

Various
* RAM profiling of trackintel. [#183](https://github.com/mie-lab/trackintel/issues/183)


