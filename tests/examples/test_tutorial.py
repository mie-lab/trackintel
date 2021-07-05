import os
import pandas as pd
import geopandas as gpd

import trackintel as ti

import subprocess
import pytest


@pytest.fixture(scope="function")
def change_test_dir():
    """Change the current working directory, run the function and change back."""
    os.chdir("./examples")
    yield
    os.chdir("..")


class Test_tutorial:
    """Tests for tutorial jupyter notebook."""

    def test_basic_tutorial(self, change_test_dir):
        """Test if the tutorial jupyter notebook runs without errors."""
        # convert the jupyter notebook to .py file
        args = [
            "jupyter",
            "nbconvert",
            "./trackintel_basic_tutorial.ipynb",
            "--output",
            "tempFile",
            "--to",
            "script",
            "--execute",
            "--ExecutePreprocessor.timeout=360",
        ]

        # check if the .py file runs without error
        subprocess.check_call(args)

        # remove the .py file
        os.remove("tempFile.py")
