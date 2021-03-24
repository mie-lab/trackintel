import os
import pandas as pd
import geopandas as gpd

import trackintel as ti

import pytest

class Test_tutorial():
    def test_basic_tutorial(self):
        os.popen("jupyter nbconvert --to script --execute --stdout .../examples/trackintel_basic_tutorial.ipynb | python3").read()
