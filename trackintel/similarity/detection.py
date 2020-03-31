# -*- coding: utf-8 -*-

def similarity_detection(method="dtw", data, trsh=None):
    """Method detects similar trajectories in a data set.
    
    INPUT:
        method: type of similarity measure (for instance only dtw)
        data: a GDF with positionfixes and tripleg_id
        treshold: trajectories wich are more similar than the treshold will be marked as similar
        
    RETURN: 
        sim: some datatype to store the indices of similar trajectories
            
            
        """
        
        
    
    n = data.loc[data['tripleg_id']].max() #number of trajectories +1 in the DataFrame
    
    if trsh is None:             #If trehshold is set, all distances are saved in to a matrix
        sim = np.ones(n,n)*np.inf 
    
    for i in range(n):
        tp1 = data.loc[data['tripleg_id']==i]
        for j in range(1,n):
            tp2 = data.loc[data['tripleg_id']==j]
            s = dtw(tp1,tp2)
        
            if s <= trsh:
                sim[i,j] = s
    
        
    return sim