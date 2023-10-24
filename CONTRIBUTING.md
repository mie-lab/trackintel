# How to contribute

Thank you for your interest in the *trackintel* development. All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

## Development
You can download the whole repository and install *trackintel* locally using `pip install .`.
For quick testing, use `trackintel.print_version()`.

Testing is done using [pytest](https://docs.pytest.org/en/latest).
Simply run the tests using `pytest` in the top-level trackintel folder.
In case you use `pipenv`, install *pytest* first (`pip install pytest`), then run *pytest* using this version: `python -m pytest`.
The use of [fixtures](https://pypi.org/project/fixtures/) for data generation (e.g., trips and trackpoints) is still an open todo.
As for now, there are some smaller datasets in the `tests` folder.

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

### Formatting
We use [black](https://github.com/psf/black) as our code formatter, run `python -m black . -l 120` in the *trackintel* folder to format your code automatically into black style. We additionally use [Flake8](https://github.com/PyCQA/flake8) checker.
Please be sure to format your code before making a pull request. If you wish, you can add pre-commit hooks for both flake8 and black to make all formatting easier.

## Coding conventions
This is a place to collect conventions we agreed upon until we find the right place in the doc for them

### Time stamps
All timestamps are timezone aware pandas `datetime64[ns, UTC]` objects. The default timezone should be `UTC` but the user should be free to use a different one if he wants. See [Issue 101](https://github.com/mie-lab/trackintel/issues/18). 

### Tests 
#### Organization of tests
See [Issue 23](https://github.com/mie-lab/trackintel/issues/23)
- The test folder copies the folder structure that the trackintel.trackintel folder has.
- Every python module has a single test file
- Every function has 1 test class
- Every method of this function should test a single property

#### Test data
If possible test data should be
- independent of unrelated preprocessing steps (e.g., avoid starting with positionfixes if you write tests for trips)
- simple and easy to understand (e.g., try to have a short example with an isolated special case rather than a large dataset that contains a lot of special cases)
- defined directly in the code itself (e.g, [this example](https://github.com/mie-lab/trackintel/blob/e0c0cdd0d8472ba7b113b3819d062ea8abcd8168/tests/io/test_postgis_gpd.py#L50)

### Integrety of input data
Functions should never change the input dataframe but rather return an altered copy.

### Adressing geometry columns
Geometry columns should never be adressed by name but by the geometry attribute [Issue 14](https://github.com/mie-lab/trackintel/issues/14), [Issue 15](https://github.com/mie-lab/trackintel/issues/15)

### Order of functions in code
The main function should be on the top of the file, the internal/secondary functions should be at the end of the file

### ID management
All trackintel objects have an ID that is the index of the dataframe [Issue 97](https://github.com/mie-lab/trackintel/issues/97)

### Docstrings
See [issue 117](https://github.com/mie-lab/trackintel/issues/117)
- All docstrings follow the [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
- The example section is mandatory.

### Performance benchmarking
- We use [airspeed velocity](https://asv.readthedocs.io/en/stable/) to benchmark key trackintel functions. 
- Benchmarks are written in the airspeed velocity format. 
- Three types of benchmarks exist (sample at [benchmarks](https://github.com/mie-lab/trackintel/tree/master/benchmarks) folder))
  - _mem_ which measures the memory of the data structure returned:
    https://github.com/mie-lab/trackintel/blob/19fcf965fce4a2bca2032f72b2759c7625c02b2f/benchmarks/preprocessing_benchmarks.py#L24

  - _peakmem_ which measure the peak memory usage:
    https://github.com/mie-lab/trackintel/blob/19fcf965fce4a2bca2032f72b2759c7625c02b2f/benchmarks/preprocessing_benchmarks.py#L27
  
  - _time_ which measure run time:
    https://github.com/mie-lab/trackintel/blob/19fcf965fce4a2bca2032f72b2759c7625c02b2f/benchmarks/preprocessing_benchmarks.py#L21
- We store the benchmark html files in the `gh-pages` branch for hosting them on the server. Detailed instructions for re-running existing benchmarks can be found in the ASV-BENCHMARKS.MD file [here](https://github.com/mie-lab/trackintel/blob/master/benchmarks/ASV-BENCHMARKING.md). 

### Others
- We limit all lines to a maximum of 120 characters.
- New release version tags use [semantic numbering](https://semver.org/).
- Commits follow the standard of [pandas](https://pandas.pydata.org/pandas-docs/stable/development/contributing.html#committing-your-code).
- There should be no warnings when running tests.
- We agreed on the following naming conventions for trackintel datatypes:
  - long: positionfixes, staypoints, triplegs, locations, trips, and tours
  - short: pfs, sp, tpls, locs, trips, tours

## Version release checklist
Before you release a new version you should check/modify the following files:

- trackintel/__ version__.py -> update the version number

- docs/conf.py -> update the version number

- check setup.py if new dependencies need greater python version

After the release is tagged, __no new release with the same version number__ can be published on pypi!

