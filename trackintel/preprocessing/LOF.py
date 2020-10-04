# -*- coding: utf-8 -*-
# @author: cleutene

from sklearn.neighbors import LocalOutlierFactor


class LOF(LocalOutlierFactor):
    def __init__(self,
                 n_neighbors=20,
                 algorithm='auto',
                 leaf_size=30,
                 contamination='auto',
                 n_jobs=None):
        novelty = True  # so that we can use fit predict seperatly
        super().__init__(n_neighbors=n_neighbors,
                         algorithm=algorithm,
                         leaf_size=leaf_size,
                         metric='minkowski',
                         p=2,
                         metric_params=None,
                         contamination=contamination,
                         novelty=novelty,
                         n_jobs=n_jobs)

    def outlier_score(self, X):
        return -self._score_samples(X)
