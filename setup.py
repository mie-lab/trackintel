#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "trackintel"
DESCRIPTION = "Human mobility and movement analysis framework."
URL = "https://github.com/mie-lab/trackintel"
EMAIL = "dobucher@ethz.ch, martinhe@ethz.ch"
AUTHOR = "Dominik Bucher, Henry Martin, Ye Hong"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None
LICENSE = "MIT"

# What packages are required for this module to be executed?
REQUIRED = [
    "pandas",
    "geopandas",
    "matplotlib",
    "numpy",
    "pint",
    "shapely",
    "networkx",
    "geoalchemy2",
    "osmnx",
    "scikit-learn",
    "tqdm",
    "similaritymeasures",
    "pygeos",
]

install_requires = [
    "pandas",
    "matplotlib",
    "numpy",
    "pint",
    "shapely",
    "networkx",
    "geoalchemy2",
    "osmnx",
    "scikit-learn",
    "tqdm",
    "geopandas",
    "similaritymeasures",
    "pygeos",
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    # $ setup.py publish support.
    cmdclass={
        # 'upload': UploadCommand,
    },
)
