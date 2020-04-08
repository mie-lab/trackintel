# -*- coding: utf-8 -*-

import trackintel as ti
import numpy as np
from ti.similarity.detection import similarity_detection

class testSimilarityDetection:
    def test_similarity_detection_trsh(self):
        
        pfs = None #some test data
        
        sim = similarity_detection(pfs, 20)
        assert sim is dict
        assert all(value<=20 for value in sim.values())
        
    
    def test_similarity_detection_no_trsh(self):
        pfs = None #some test data
        a = 1 # id of a specific tripleg
        b = 2 # id of a specific trioleg
        
        sim = similarity_detection(pfs)
        assert isinstance(sim, np.ndarray)
        assert sim()
    
        