# -*- coding: utf-8 -*-

from scipy.sparse import dok_matrix

def similarity_detection(method="dtw", data, trsh=None):
    """Method detects similar trajectories in a data set.
    
    INPUT:
        method:        type of similarity measure (for instance only dtw)
        data:       a GDF with positionfixes and tripleg_id
        treshold:   trajectories wich are more similar than the treshold will be marked as similar
                    if treshold is none, a distance matrix of all trajectories will be calculated
        
    RETURN: 
        sim: 
            case treshold is set: Dictionary with tripleg indices as keys and the distance of the trajectories as values
            {(tp1_id,tp2_id):dist_tp1_tp2}, only distances under the treshold are stored
            
            case treshold is none: Distance matrix of all trajectories
            
            
        """
    try:
        assert data.as_positionfixes._validate()
        assert 'tripleg_id' in data.columns
    except: print('Input data format must be positionfixes with added tripleg_id')
        
    
    n = data.loc[data['tripleg_id']].max() #number of trajectories +1 in the DataFrame
    
    if trsh is None:             #If trehshold is not set, all distances are saved in to a matrix
        sim = np.ones(n,n)*np.inf 
    
        for i in range(n):
            tp1 = data.loc[data['tripleg_id']==i]
            for j in range(1,n):
                tp2 = data.loc[data['tripleg_id']==j]
                s = dtw(tp1,tp2)
                sim[i,j] = s
    else:
        sim = {}
        for i in range(n):
            tp1 = data.loc[data['tripleg_id']==i]
            for j in range(1,n):
                tp2 = data.loc[data['tripleg_id']==j]
                s = dtw(tp1,tp2)
                
                if s <= trsh:
                    sim.update({(i,j):s})
    
        
    return sim