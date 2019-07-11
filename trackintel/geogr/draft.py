# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:37:36 2019

@author: martinhe
"""
import numpy as np

def dist(a1,a2,b1,b2):
    z = []
    
    for i in range(len(a1)):
        zz = int(str(a1[i])+str(a2[i])+str(b1[i])+str(b2[i]))    
        z.append(zz)
    
    return z

x1_in = np.arange(1,n+1)
x2_in = np.arange(1,n+1)
y1_in = np.arange(1,n+1)
y2_in = np.arange(1,n+1)


x_ix, y_ix = np.triu_indices(n)    


x1 = x1_in[x_ix]
x2 = x2_in[x_ix]
y1 = y1_in[y_ix]
y2 = y2_in[y_ix]

d = dist(x1,x2,y1,y2)

# rebuild matrix from vector
D = np.zeros((n,n))

k = 0
for i in range(n):
    for j in np.arange(i+1,n):
        D[i,j] = x[i]
        k = k+1
        