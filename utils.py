import numpy as np

def normTT(x):
    min1 = 1 #min(Tout) # min for ambient temperature
    max1 = 96 #max(Tout) # max for ambient temperature
    x_scaled = np.ones((x.shape))
    x_scaled = (x-min1)/(max1-min1)
    return x_scaled

def norm01(x):
    min1 = min(x)
    max1 = max(x) 
    x_scaled = np.ones((x.shape))
    x_scaled = (x-min1)/(max1-min1)
    return x_scaled

def normp(x,y):
    x_scaled = np.zeros(x.shape)
    w = 0
    for q in range(len(x_scaled)):
        if q%y==0:
            maxx = max(x[w*y:(w+1)*y])
            w = w+1
        x_scaled[q] = x[q]/maxx
    return x_scaled
