# The trackintel Framework

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mie-lab/trackintel/master?filepath=%2Fexamples%2Fexample_geolife%2FTrackintel_introduction.ipynb)
[![PyPI version](https://badge.fury.io/py/trackintel.svg)](https://badge.fury.io/py/trackintel)
[![Build Status](https://travis-ci.org/mie-lab/trackintel.svg?branch=master)](https://travis-ci.org/mie-lab/trackintel)
[![Documentation Status](https://readthedocs.org/projects/trackintel/badge/?version=latest)](https://trackintel.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/gh/mie-lab/trackintel/coverage.svg?branch=master)](https://codecov.io/gh/mie-lab/trackintel)
          
*trackintel* is a library for the analysis of spatio-temporal tracking data with a focus on human mobility. The core of *trackintel* is the hierachical data model for movement data that is used in transport planning [[1]](#1). We provide functionalities for the full life-cycle of human mobility data analysis: import and export of tracking data of different types (e.g, trackpoints, check-ins, trajectories, etc.), preprocessing, data quality assessment, semantic enrichment, quantitative analysis and mining tasks, and visualization of data and results.
Trackintel is based on [Pandas](https://pandas.pydata.org/) and [GeoPandas](https://geopandas.org/#)

You can find the documentation on the [trackintel documentation page](https://trackintel.readthedocs.io/en/latest).

## Data model

An overview of the data model of *trackintel*:
* **positionfixes** (raw tracking points, e.g., GPS)
* **staypoints** (locations where a user spent time without moving, e.g., aggregations of positionfixes or check-ins)
* **activities** (staypoints with a purpose and a semantic label, e.g., meeting to drink a coffee as opposed to waiting for the bus)
* **locations** (important places that are visited more than once)
* **triplegs** (or stages) (continuous movement without changing mode, vehicle or stopping for too long, e.g., a taxi trip between pick-up and drop-off)
* **trips** (The sequence of all triplegs between two consecutive activities)
* **tours** (A collection of sequential trips that return to the same location)

You can enter the trackintel framework if your data corresponds to any of the above mentioned movement data representation. Here are some of the functionalities that we provide: 
* **Import**: Import from the follwoing data formats is supported `geopandas dataframes` (recommended), `csv files` in a specified format, `postGIS` databases, and we have specific dataset readers for popular public datasets (e.g, geolife).
* **Aggregation**: We provide functionalities to aggregate into the next level of our data model. E.g., positionfixes->staypoints; positionfixes->triplegs; staypoints; staypoints->locations; staypoints+triplegs->trips; trips->tours
* **Enrichment**: Activity semantics for staypoints; Mode of transport semantics for triplegs; High level semantics for locations

## Installation and Usage
*trackintel* is on [pypi.org](https://pypi.org/project/trackintel/), you can install it with `pip install trackintel` as long as `GeoPandas` is already installed. 

You should then be able to run the examples in the `examples` folder or import trackintel using:
```{python}
import trackintel
```

## Development
You can install *trackintel* locally using `pip install .`.
For quick testing, use `trackintel.print_version()`.

Testing is done using [pytest](https://docs.pytest.org/en/latest).
Simply run the tests using `pytest` in the top-level trackintel folder.
In case you use `pipenv`, install *pytest* first (`pip install pytest`), then run *pytest* using this version: `python -m pytest`.
The use of [fixtures](https://pypi.org/project/fixtures/) for data generation (e.g., trips and trackpoints) is still an open todo.
As for now, there are some smaller datasets in the `tests` folder.

Versions use [semantic numbering](https://semver.org/).
Commits follow the standard of [Conventional Commits](https://www.conventionalcommits.org).
You can generate them easily using [Commitizen](https://github.com/commitizen/cz-cli).

You can find the development roadmap under `ROADMAP.md`.

### Documentation

The documentation follws the [pandas resp. numpy docstring standard](https://pandas-docs.github.io/pandas-docs-travis/development/contributing.html#contributing-to-the-documentation).
In particular, it uses [Sphinx](http://www.sphinx-doc.org/en/master/) to create the documentation.
You can install Sphinx using `pip install -U sphinx` or `conda install sphinx`.

If you use additional dependencies during development, do not forget to add them to `autodoc_mock_imports` in `docs/conf.py` for readthedocs.org to work properly.

You can then generate the documentation using `sphinx-build -b html docs docs.gen`.
This will put the documentation in `docs.gen`, which is in `.gitignore`.

### Continuous Integration

There are travis and appveyor CIs set up for Unix/Windows builds.
You can find the corresponding scripts in `.travis.yml` and `appveyor.yml`.
Adding [Coveralls](https://coveralls.io) is an open todo.

## Contributors

trackintel is primarily maintained by the Mobility Information Engineering Lab at ETH Zurich ([mie-lab.ethz.ch](http://mie-lab.ethz.ch)).
If you want to contribute, send a pull request and put yourself in the `AUTHORS.md` file.

## References
<a id="1">[1]</a>
[Axhausen, K. W. (2007). Definition Of Movement and Activity For Transport Modelling. In Handbook of Transport Modelling. Emerald Group Publishing Limited.](
https://www.researchgate.net/publication/251791517_Definition_of_movement_and_activity_for_transport_modelling)
