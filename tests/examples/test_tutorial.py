import os
import pandas as pd
import geopandas as gpd

import trackintel as ti

import pytest

@pytest.fixture(scope="function")
def change_test_dir():
    os.chdir('./examples')
    yield
    os.chdir('..')

class Test_tutorial():
    def test_basic_tutorial(self, change_test_dir):
        """
        Tests if the tutorial jupyter notebook runs without errors

        Returns
        -------
        None.

        """
  
        os.popen("jupyter nbconvert --to script --execute --stdout ./trackintel_basic_tutorial.ipynb | python3").read()
        #os.remove('./trackintel_basic_tutorial.py')

