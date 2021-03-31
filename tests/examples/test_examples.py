#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

class Test_examples():
    def test_example_preprocess_trajectories(self):
        exec(open(os.path.join('examples','preprocess_trajectories.py')).read())
