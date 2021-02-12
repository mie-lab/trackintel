# How to contribute
This is a place to collect conventions we agreed upon until we find the right place in the doc for them

## Coding conventions
### Time stamps
All timestamps are timezone aware pandas `datetime64[ns, UTC]` objects. The default timezone should be `UTC` but the user should be free to use a different one if he wants. See [Issue 101](https://github.com/mie-lab/trackintel/issues/18). 

### Organization of tests
See [Issue 23](https://github.com/mie-lab/trackintel/issues/23)
- The test folder copies the folder structure that the trackintel.trackintel folder has.
- Every python module has a single test file
- Every function has 1 test class
- Every method of this function should test a single property

### Warnings
There should be no warnings when running tests.

### Integrety of input data
Functions should never change the input dataframe but rather return an altered copy.

### Adressing geometry columns
Geometry columns should never be adressed by name but by the geometry attribute [Issue 14](https://github.com/mie-lab/trackintel/issues/14), [Issue 15](https://github.com/mie-lab/trackintel/issues/15)

### Order of functions in code
The main function should be on the top of the file, the internal/secondary functions should be at the end of the file

### ID management
All trackintel objects have an ID that is the index of the dataframe [Issue 97](https://github.com/mie-lab/trackintel/issues/97)

### Others
- We limit all lines to a maximum of 120 characters.

## Version release checklist
Before you release a new version you should check/modify the following files:

- trackintel/__ version__.py -> update the version number

- conf.py -> update the version number

- check setup.py if new dependencies need greater python version

After the release is tagged, __no new release with the same version number__ can be published on pypi!

