# Roadmap

This document explains the functionality planned for various releases.

## v1.0.0

Steps until we get to version 1.0.0.

### v0.1.0

* :heavy_check_mark: Read *positionfixes*, *triplegs*, *staypoints*, *places*, *trips* from CSV files.
* :heavy_check_mark: Read *positionfixes*, *triplegs*, *staypoints*, *places*, *trips* from PostGIS.
* :heavy_check_mark: Write *positionfixes*, *triplegs*, *staypoints*, *places*, *trips* to CSV files.
* :heavy_check_mark: Write *positionfixes*, *triplegs*, *staypoints*, *places*, *trips* to PostGIS.
* :heavy_check_mark: Visualize *positionfixes*, *triplegs*, *staypoints* geographically.

### v0.2.0

* :heavy_check_mark: Extract *triplegs* and *staypoints* from *positionfixes*.
* Functions for smoothening tripleg data (e.g., Douglas-Peucker).

### v0.3.0

* Extract *trips* and *places* from *triplegs* and *staypoints*.
* Augment model of *triplegs* and *staypoints* in such a way that they can
  hold references to *trips* and *places*.
* Visualize *trips* and *places* geographically, e.g., by coloring *triplegs*
  differently or by making a buffer around all *staypoints* that are part of
  a *place*.

### v0.4.0

* :heavy_check_mark: Create transition graphs from *places*.
* :heavy_check_mark: Provide and visualize a range of (transition) graph measures.
* Add context data to movement trajectories (using spatio-temporal map algebra).

### v0.5.0

* Impute resp. infer transport modes. This functionality is based on properties of the movement data itself (speed, associated features from accelerometer data, etc.) as well as by aligning them with context data such as from OpenStreetMap or GTFS departure schedules. 
* Provide holistic transport mode inferral (take into account the fact that if someone uses the car to reach a certain location, the person is likely to leave by car as well).

### v0.6.0

* Map match triplegs (based on transport mode identification). 
* Tripleg imputation: when someone "suddenly" appears at another location, there must be a transition in between.

### v0.7.0

* Anomaly detection (based on properties of movement data as well as contextual factors).
* Data quality assessments: statistical measures and visualization (e.g., of tracking frequency, both spatial and temporal).

### v0.8.0

* Activity inference: what did someone likely do at a certain location?

### v0.9.0

* Mobility behavior (and its changes): automatic detection of uncharacteristic changes in mobility patterns.
* User profiling and clustering.

### Final Steps for v1.0.0

* Documentation and examples.

## v2.0.0

Ideas for a later version.

Augment tracking data:

* Short walks to/from cars/buses/etc.: These are often not recognized by the tracking applications.

Implement *tours* (starting and ending at a persons home location) and *customermovements* (consecutive triplegs with using transport provided by a single provider (e.g., a local bus company)).

* Read *customermovements*, *tours* from files.
* Read *customermovements*, *tours* from PostGIS.
* Write *customermovements*, *tours* to files.
* Write *customermovements*, *tours* to PostGIS.
* Visualize *customermovements*, *tours* geographically.
* Extract *customermovements* from *triplegs* and *trips*.
* Extract *tours* from *trips*.
