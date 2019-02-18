# The trackintel Framework

![Version](https://img.shields.io/badge/version-v0.0.1-red.svg)

Focusing on human mobility data, *trackintel* provides functionalities for data quality enhancement, integrating data from various sources, performing quantitative analysis and mining tasks, and visualizing the data and/or analysis results. In addition to these core functionalities, packages are provided for user mobility profiling and trajectory-based learning analytics.

You can find the documentation under [the Wiki](https://github.com/mie-lab/trackintel/wiki) and on the [trackintel documentation page](https://mie-lab.github.io/trackintel-docs) (the Wiki is to be discontinued).

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

## Installation and Usage

This is not on [pypi.org](https://pypi.org/) yet, so to install you have to `git clone` the repository and install it with `pip install .`.

You should then be able to run the examples in the `examples` folder or import trackintel using:
```{python}
import trackintel
```

## Development

You can install *trackintel* locally using `pip install .`. For quick testing, use `trackintel.print_version()`.

Testing is done using [pytest](https://docs.pytest.org/en/latest). Simply run the tests using `pytest` in the top-level trackintel folder.

### Documentation

The documentation follws the [pandas resp. numpy docstring standard](https://pandas-docs.github.io/pandas-docs-travis/development/contributing.html#contributing-to-the-documentation). In particular, it uses [Sphinx](http://www.sphinx-doc.org/en/master/) to create the documentation.

You can install Sphinx using `pip install -U sphinx` or `conda install sphinx`. Generate the documentation using `python make.py` in the `doc` folder. Attention! This will copy the generated HTML to a folder `trackintel-docs` (containing the documentation repository [https://github.com/mie-lab/trackintel-docs](https://github.com/mie-lab/trackintel-docs)), which must exist in parallel to this repository. After copying, you can push the other repository to update the documentation.

## Contributors

trackintel is primarily maintained by the Mobility Information Engineering Lab at ETH Zurich ([mie-lab.ethz.ch](http://mie-lab.ethz.ch)). If you want to contribute, send a pull request and put yourself in the `AUTHORS.md` file.