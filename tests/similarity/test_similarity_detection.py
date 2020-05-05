# -*- coding: utf-8 -*-

import trackintel as ti
import numpy as np
from trackintel.similarity.detection import similarity_detection
import os
import pytest
from scipy import sparse

class testSimilarityDetection:
    pfs = ti.io.file.read_positionfixes_csv(os.path.join('tests','data','tplset_as_pfs.csv'))
    pfs.crs='EPSG:4326'
    pfs = pfs.to_crs(epsg=3395)
    def test_similarity_detection_general(self):
        false_data = ti.io.read_positionfixes_csv(os.path.join('tests','data','positionfixes.csv'))
        with pytest.raises(Exception):
            similarity_detection(false_data)
            
        tp1 = self.pfs[self.pfs['tripleg_id']==1]
        tp2 = self.pfs[self.pfs['tripleg_id']==2]
        sim = similarity_detection(self.pfs,'dtw')
        assert sim[1,2]==ti.similarity.measures.e_dtw(tp1,tp2)
    
    def test_similarity_detection_trsh(self):
        sim = similarity_detection(pfs, 'dtw', 20)
        assert isinstance(sim, sparse.dok.dok_matrix)
        assert all(value<=20 for value in sim.values())
        
    
    def test_similarity_detection_no_trsh(self):
        self.pfs = None #some test data
        a = 1 # id of a specific tripleg
        b = 2 # id of a specific trioleg
        
        sim = similarity_detection(self.pfs)
#        assert sim[a,b]== #some value
        assert (sim == sim.transpose()).all()
    
    def test_similarity_detection_dist(self):
        sim = similarity_detection(self.pfs,'dtw', dist=True)
        assert isinstance(sim, np.array)
        