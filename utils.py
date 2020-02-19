import numpy as np

def hour_to_quarter(Tout24,p24,st):
    Tout = np.ones((st*len(Tout24),1))
    price = np.ones((st*len(p24),1))
    w = 0
    for q in range(len(Tout)):
        if q%st==0:
            Tout[q] = Tout24[w]
            price[q] = p24[w]
            w = w+1
        else:
            Tout[q] = Tout24[w-1]
            price[q] = p24[w-1]
            
    return Tout, price

def compute_cost_reduction(price,uphys,uphys_base,st,umax,T):
    daily_cost_baseline = np.sum(price[-T:]/st*10*uphys_base[-T:]*umax/1000)
    daily_cost_dem = np.sum(price[-T:]/st*10*uphys[-T:]*umax/1000)
    daily_cost_reduction = 100-daily_cost_dem/daily_cost_baseline*100
    
    cum_cost_baseline = np.sum(price/st*10*uphys_base*umax/1000)
    cum_cost_dem = np.sum(price/st*10*uphys*umax/1000)
    cum_cost_reduction = 100-cum_cost_dem/cum_cost_baseline*100
   
    return daily_cost_reduction, cum_cost_reduction

def compute_baseline(Tin0,Tm0,action,Tout,umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b):
    Tin = np.zeros((len(Tout)+1,1))
    Tm = np.zeros((len(Tout)+1,1))
    uphys = np.zeros((len(Tout),1))
    Tin[0] = Tin0
    Tm[0] = Tm0
    uphys[0] = action
    for q in range(len(Tout)):            
        if Tin[q]<Tin_min:
            uphys[q] = 1
        if Tin[q]>Tin_max:
            uphys[q] = 0
        Tin[q+1] = a*Tin[q]+(1-a)*Ra/(Ra+Rm)*Tm[q]+(1-a)*(Rm/(Ra+Rm)*Tout[q]+Ra*Rm/(Ra+Rm)*Qheat*uphys[q])
        Tm[q+1] = b*Tm[q]+(1-b)*Tin[q]
        if q<len(Tout)-1:
            uphys[q+1] = uphys[q]

    return Tin, Tm, uphys

def true_step(Tin,Tm,action,Tout,umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b):     
    if Tin<Tin_min:
        action = 1
    if Tin>Tin_max:
        action = 0
    Tin_next = a*Tin+(1-a)*Ra/(Ra+Rm)*Tm+(1-a)*(Rm/(Ra+Rm)*Tout+Ra*Rm/(Ra+Rm)*Qheat*action)
    Tm_next = b*Tm+(1-b)*Tin

    return Tin_next, Tm_next, action
