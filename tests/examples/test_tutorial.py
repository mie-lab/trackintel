import os
import pandas as pd
import geopandas as gpd

import trackintel as ti

import pytest

class Test_tutorial():
    def test_basic_tutorial(self):
        """
        Tests if the tutorial jupyter notebook runs without errors

        Returns
        -------
        None.

        """
        
        os.popen("jupyter nbconvert --to script --execute --stdout ./examples/trackintel_basic_tutorial.ipynb | python3").read()
        os.remove('./examples/trackintel_basic_tutorial.py')
