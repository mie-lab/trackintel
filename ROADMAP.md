# Roadmap

This document explains the functionality planned for various releases.

## v1.0.0

Steps until we get to version 1.0.0.

### v0.1.0

* Read *positionfixes*, *triplegs*, *staypoints* from CSV files.
* Read *positionfixes*, *triplegs*, *staypoints* from PostGIS.
* Write *positionfixes*, *triplegs*, *staypoints* to CSV files.
* Write *positionfixes*, *triplegs*, *staypoints* to PostGIS.
* Visualize *positionfixes*, *triplegs*, *staypoints* geographically.

### v0.2.0

* Extract *triplegs* and *staypoints* from *positionfixes*.

### v0.3.0

* Extract *trips* and *places* from *triplegs* and *staypoints*.
* Augment model of *triplegs* and *staypoints* in such a way that they can
  hold references to *trips* and *places*.
* Visualize *trips* and *places* geographically, e.g., by coloring *triplegs*
  differently or by making a buffer around all *staypoints* that are part of
  a *place*.

### v0.4.0

* Read *customermovements*, *tours* from files.
* Read *customermovements*, *tours* from PostGIS.
* Write *customermovements*, *tours* to files.
* Write *customermovements*, *tours* to PostGIS.
* Visualize *customermovements*, *tours* geographically.

### v0.5.0

* Extract *customermovements* from *triplegs* and *trips*.
* Extract *tours* from *trips*.

### Further Down the Road: v0.6.0 - v1.0.0

These points will be assigned to respective version numbers soon.

* Impute and infer transport modes.
* Align transport modes.
* Data quality indicators.
* Detect anomalies.
* Map matching.
* Impute missing triplegs.
* Infer activities.
* Add context data.
* Analyze and detect mobility behavior.
* Short walks.
* User profiling and clustering.