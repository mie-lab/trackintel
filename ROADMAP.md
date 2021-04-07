# Roadmap

This document explains the functionality planned for various releases.

## v1.0.0

Steps until we get to version 1.0.0.

### v0.1.0

* :heavy_check_mark: Read *positionfixes*, *triplegs*, *staypoints*, *locations*, *trips* from CSV files.
* :heavy_check_mark: Read *positionfixes*, *triplegs*, *staypoints*, *locations*, *trips* from PostGIS.
* :heavy_check_mark: Write *positionfixes*, *triplegs*, *staypoints*, *locations*, *trips* to CSV files.
* :heavy_check_mark: Write *positionfixes*, *triplegs*, *staypoints*, *locations*, *trips* to PostGIS.
* :heavy_check_mark: Visualize *positionfixes*, *triplegs*, *staypoints* geographically.

### v0.2.0

* :heavy_check_mark: Extract *triplegs* and *staypoints* from *positionfixes*.
* :heavy_check_mark:Functions for smoothening tripleg data (e.g., Douglas-Peucker).

### v0.3.0

* :heavy_check_mark: Extract *trips* from *triplegs* and *staypoints*
* :heavy_check_mark: Extract *locations* from *staypoints*.

### v0.4.0

* :heavy_check_mark: Create transition graphs from *locations*.
* :heavy_check_mark: Provide and visualize a range of (transition) graph measures.

### v0.5.0

* :heavy_check_mark: Augment model of *triplegs* and *staypoints* in such a way that they can
  hold references to *trips* and *locations*.

### v0.6.0
* :heavy_check_mark: Support flexible geometry columns [Issue15](https://github.com/mie-lab/trackintel/issues/15)
* :heavy_check_mark: Filter data by geographic location (e.g., all data within Switzerland)
* :heavy_check_mark: Enhance tripleg extraction to be _gap aware_ [issue27](https://github.com/mie-lab/trackintel/issues/27)

### v0.7.0

* :heavy_check_mark: Simple transport mode prediction. This functionality is based on properties of the movement data only.
* Simple activity inference. This functionality is based on properties of the movement data only and should include {Home, Work, Other}.

### v0.8.0

* :heavy_check_mark: Data quality assessments: statistical measures of temporal tracking frequency.

### v0.9.0

* :heavy_check_mark: Tutorial for trackintel

### Final Steps for v1.0.0

* Documentation
  * :heavy_check_mark: Link a guide to the readme how to contribute to the documentation
  * :heavy_check_mark: The documentation should be complete (all functions and methods are mentioned)
  * The documentation should not be focused on PostGIS 


## v2.0.0

Ideas for a later version.

I/O
* Read *customermovements*, *tours* from files.
* Read *customermovements*, *tours* from PostGIS.
* Write *customermovements*, *tours* to files.
* Write *customermovements*, *tours* to PostGIS.
* Include data input/output methods for more common datasets


Preprocessing
* Implement *tours* (starting and ending at a persons home location) and *customermovements* (consecutive triplegs with using transport provided by a single provider (e.g., a local bus company)).
* Short walks to/from cars/buses/etc.: These are often not recognized by the tracking applications.
* Map match *triplegs* (based on transport mode identification). 
* Imputation of trivial gaps in tracking data (*triplegs* and *staypoints*).
* Simple outlier filtering for *triplegs*

Augment tracking data:
* Better transport mode prediction based on movement data (speed, associated features from accelerometer data, etc.) as well as by aligning them with context data such as from OpenStreetMap or GTFS departure schedules. 
* Add context data to movement trajectories (using spatio-temporal map algebra).
* Better activity prediction. This functionality is based on properties of the movement data and on context data such as points of interests.
* Provide holistic inference of transport modes (take into account the fact that if someone uses the car to reach a certain location, the person is likely to leave by car as well).

Analysis
* Mobility behaviour (and its changes): automatic detection of uncharacteristic changes in mobility patterns.
* User profiling and clustering.
* Anomaly detection (based on properties of movement data as well as contextual factors).
* Extract *customermovements* from *triplegs* and *trips*.
* next place prediction (e.g., markov-model)
* Anomaly detection (based on properties of movement data as well as contextual factors).
* Clustering of triplegs using similarity metrics
* Include the calculation of common mobility indicators (e.g., radius of gyration)

Visualization 
* Visualize *customermovements*, *tours* geographically.
* Visualize *trips* and *locations* geographically, e.g., by coloring *triplegs*
  differently or by making a buffer around all *staypoints* that are part of
  a *locations*.

Various
* Parallelization



