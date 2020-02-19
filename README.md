# Deep-Energy-Manager
## Overview
Deep Energy Manager (DEM) is a model-free reinforcement learning (RL) agent for optimal building energy management. 
## Table of content
1. `agent.py`: DQN agent
2. `env.py`: Environment
3. `utils.py`: Utility functions
4. `run.py`: Train/test DQN agent
5. `data /`: Hourly outdoor temperature and electricity price from 12/1/2019 to 1/31/2020. Day ahead electricity price is available at https://apps.coned.com/CEMyAccount/csol/MscDayAheadCC.aspx
6. `results /`: Cost reduction: daily + complete season
## How to run

## Performance
DEM Day 1 control policy:
![alt text](https://github.com/ab5163/Deep-Energy-Manager/blob/master/results/Day%201.png)
DEM performance over 5 days:
![alt text](https://github.com/ab5163/Deep-Energy-Manager/blob/master/results/Day%201.png)
## Acknowledgements
DQN agent is inspired by 
* @keon
* @Khev
