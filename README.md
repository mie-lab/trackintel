# The trackintel Framework

![Version](https://img.shields.io/badge/version-v0.0.1-red.svg)

Focusing on human mobility data, *trackintel* provides functionalities for data quality enhancement, integrating data from various sources, performing quantitative analysis and mining tasks, and visualizing the data and/or analysis results. In addition to these core functionalities, packages are provided for user mobility profiling and trajectory-based learning analytics.

You can find the documentation under [the Wiki](https://github.com/mie-lab/trackintel/wiki).

## Target Users

*trackintel* is intended for use mainly by researchers with:

* Programming experience in Python
* Proficiency in movement data mining and analysis

## Assumptions

* Movement data exists in csv, (geo)json, gpx or PostGIS format
* Movement data consists of points with x,y-coordinates, a time stamp, an optional accuracy and a user ID
* The tracking data can be reasonably segmented into 
  * positionfixes (raw tracking points)
  * triplegs (aggregated tracking points based on the transport mode)
  * trips (aggregated activities based on the visited destination / staypoint)
  * tours (aggregated trips starting / ending at the same location / staypoint)
* One of the following transportation modes was used at any time: car, walking, bike, bus, tram, train, plane, ship, e-car, e-bike

## Development

You can install *trackintel* locally using `pip install .`. For quick testing, use `trackintel.print_version()`.
