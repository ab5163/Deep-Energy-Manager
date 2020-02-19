import random
import numpy as np
import time
import os
from numpy.random import seed
seed(7) # fix random seed for reproducibility
import pandas as pd
import math
from pytictoc import TicToc
import matplotlib.pyplot as plt
import pylab as pl
np.set_printoptions(precision=2)
from decimal import *
getcontext().prec = 2
from agent import Agent
from BuildingEnv import Env
from utils import hour_to_quarter, compute_cost_reduction, compute_baseline, true_step

t = TicToc()
t1 = TicToc()
t2 = TicToc()
t3 = TicToc()
t4 = TicToc()
t5 = TicToc()
t6 = TicToc()
t7 = TicToc()

# Data & settings

umax = 2300 # max power in W
SCOP = 3.0 # Seasonal COP
Qheat = umax*SCOP
Tin_min = 20
Tin_max = 23
st = 4
T = 24*st
Ra = 1/125 #  equivalent thermal resistance for building air [degC/W]
Rm = 1/6863 #  equivalent thermal resistance for building mass [degC/W]
Ca = 2.441e6 # equivalent heat capacity for building air [J/degC]
Cm = 9e6 # equivalent heat capacity for building mass [J/degC]
dt = 60*60/st # control time step [s]
a = math.exp(-(Ra+Rm)/(Ra*Rm*Ca)*dt)
b = math.exp(-1/(Rm*Cm)*dt)

Data = pd.read_csv('/scratch/ab5163/RL/Space heating/12-2017 1-2018.csv')
Hist = pd.read_csv('/scratch/ab5163/RL/Space heating/11-2017.csv')
Data = Data.values
Hist = Hist.values
p24 = Data[:,1]
Tout24 = Data[:,2]
hist_T24 = Hist[:,0]
hist_p24 = Hist[:,1]
hist_Tout24 = Hist[:,2]

# Converting hours to quarters


Tout, price = hour_to_quarter(np.concatenate((hist_Tout24,Tout24)),np.concatenate((hist_p24,p24)),st)

# Defining parameters
SL = int(len(Tout)/T) # Season length
F = 2
daily_cost_reduction = [] # Cumulative cost reduction
cum_cost_reduction = [] # Cumulative cost reduction
sensors = True

# DQN agent settings
num_states, num_actions = 3, 2
lr, gamma, tau, buffer_size = 1e-5, 1, 0.01, 500000
l1_units, l2_units, l3_units = 256, 512, 256
learning_start, rnd_seed = 100, 7
random.seed(rnd_seed)
agent = Agent(num_states, num_actions, lr, gamma, tau, buffer_size, l1_units, l2_units, l3_units, rnd_seed)
EPISODES = int(1e4)
progress = 100 # agent progress report frequency
score = np.zeros((int(EPISODES/progress),SL-1))

# First day is used to generate samples for training the building environment, control policy is the baseline control
Tin0 = random.uniform(20, 23)
Tm0 = random.uniform(Tin0-0.1,Tin0+0.1)
action = random.randrange(2)
Tin, Tm, uphys = compute_baseline(Tin0,Tm0,action,Tout[:F*T],umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b)
Tin_env = np.copy(Tin)
u = np.copy(uphys)
uphys_base = np.copy(uphys)
for q in range(F):
    dcr, ccr = compute_cost_reduction(price[q*T:(q+1)*T],uphys[q*T:(q+1)*T],uphys_base[q*T:(q+1)*T],st,umax,T)
    daily_cost_reduction.append(dcr)
    cum_cost_reduction.append(ccr)
    print("Day: {}, Daily cost reduction: {}, Cumulative cost reduction: {}".format(q+1, dcr, ccr))

t.tic()
for B in range(F,SL):
    t1.tic()
    Tin_base0, Tm_base0, uphys_base0 = compute_baseline(Tin[-1],Tm[-1],uphys[-1],Tout[B*T:(B+1)*T],umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b)
    uphys_base = np.concatenate((uphys_base, uphys_base0))
    daily_price = price[B*T:(B+1)*T]
    N, P = 10, np.sum(uphys_base0)
    env = Env(Tout[:(B+1)*T], daily_price, Tin_min, Tin_max, T, F, u, uphys, rnd_seed, N, P)
    env.train_models(sensors, Tin[-1])
    Tin = np.concatenate((Tin,np.zeros((T,1))))
    Tm = np.concatenate((Tm,np.zeros((T,1))))
    Tin_env = np.concatenate((Tin_env,np.zeros((T,1))))
    uphys = np.concatenate((uphys,np.zeros((T,1))))
    u = np.concatenate((u,np.zeros((T,1))))
    # training loop
    t2.tic()
    ii = 0
    for episode in range(1,EPISODES+1):
        if episode % progress == 1:
            t3.tic()
        state, done = env.reset()
        Qvalues = np.zeros((T,2))
        step = 0

        while not done:

            if episode<=learning_start:
                action = random.randrange(2)
            else:
                Qvalues[step] = agent.act(state)
                action = np.argmax(Qvalues[step])

            next_state, reward, done = env.take_step(action)
            next_step = step+1           
            agent.remember(state, action, reward, next_state, done)
            if episode>=learning_start:
                agent.replay(batch_size=32)
                agent.target_train()
            state = next_state
            step = next_step
        
        if episode % progress == 0 and episode > learning_start:           
               
            soc = env.soc_true 
            for e in range(T):

                state = np.column_stack((e/T,soc,env.Tamb[e]))

                Qvalues[e] = agent.act(state)
                action = np.argmax(Qvalues[e])
                u[B*T+e] = action
                uphys[B*T+e] = action

                soc = env.take_eval_step(soc,action,e)
                Tin_env[B*T+e+1] = Tin_min+soc*(Tin_max-Tin_min)
                Tin[B*T+e+1], Tm[B*T+e+1], uphys[B*T+e] = true_step(Tin[B*T+e],Tm[B*T+e],uphys[B*T+e],Tout[B*T+e],umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b)

            dcr, ccr = compute_cost_reduction(price[:(B+1)*T],uphys[:(B+1)*T],uphys_base[:(B+1)*T],st,umax,T)

            print("Day: {}, Episode: {}/{}, cost_reduction: {}".format(B+1, episode, EPISODES, round(dcr)))
            print(np.column_stack((env.price,u[B*T:(B+1)*T],uphys[B*T:(B+1)*T],Tin[B*T:(B+1)*T],Tin_env[B*T:(B+1)*T])))
            score[ii,B-1] = dcr
            ii += 1
            t3.toc()
    t2.toc() 
    # Control policy for next day
    soc = env.soc_true 
    for e in range(T):

        state = np.column_stack((e/T,soc,env.Tamb[e]))

        Qvalues[e] = agent.act(state)
        action = np.argmax(Qvalues[e])
        u[B*T+e] = action
        uphys[B*T+e] = action
        soc = env.take_eval_step(soc,action,e)
        Tin_env[B*T+e+1] = Tin_min+soc*(Tin_max-Tin_min)
        Tin[B*T+e+1], Tm[B*T+e+1], uphys[B*T+e] = true_step(Tin[B*T+e],Tm[B*T+e],uphys[B*T+e],Tout[B*T+e],umax,SCOP,Qheat,Tin_min,Tin_max,st,T,Ra,Rm,Ca,Cm,dt,a,b)
        if sensors:
            soc = (Tin[B*T+e+1]-Tin_min)/(Tin_max-Tin_min)


    dcr, ccr = compute_cost_reduction(price[:(B+1)*T],uphys[:(B+1)*T],uphys_base[:(B+1)*T],st,umax,T)
    daily_cost_reduction.append(dcr)
    cum_cost_reduction.append(ccr)

    print("Day: {}, Daily cost reduction: {}, Cumulative cost reduction: {}".format(B+1, round(dcr), round(ccr)))
    print(np.column_stack((env.price,u[B*T:(B+1)*T],uphys[B*T:(B+1)*T],Tin[B*T:(B+1)*T],Tin_env[B*T:(B+1)*T])))      
    t1.toc()
    agent.reset_weights(True)
t.toc()
