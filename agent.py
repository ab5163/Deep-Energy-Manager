import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

class Agent:
    
    def __init__(self, input_dim, output_dim, lr, gamma, tau, buffer_size, l1_units, l2_units, l3_units, learning_start):

        self.memory  = deque(maxlen=buffer_size)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.ls = learning_start
        self.epsilon_const = self.ls+1000
        self.epi = 0
        self.learning_rate = lr
        self.tau = tau
        self.l1_units = l1_units
        self.l2_units = l2_units
        self.l3_units = l3_units

        self.model, self.init_weights = self.create_model()
        self.target_model, self.target_init_weights = self.create_model()

    def xplr(self):
        self.epsilon = (self.epsilon_min-1)/self.epsilon_const*self.epi+1
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.epi += 1
    
    def create_model(self):
        model   = Sequential()
        model.add(Dense(self.l1_units, input_dim = self.input_dim, activation="relu"))
        model.add(Dense(self.l2_units, activation="relu"))
        model.add(Dense(self.l3_units, activation="relu"))
        model.add(Dense(self.output_dim))
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
        init_weights = model.get_weights()
        return model, init_weights

    def act(self, state):
        self.xplr()
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return self.model.predict(state)[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        states = []
        targets = []
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state[0])
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward/100
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = np.clip(reward/100 + Q_future * self.gamma, -1, 0)
            targets.append(target[0])
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states,targets,verbose=False, epochs=1)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        
    def reset_weights(self, reset_weights):
        if reset_weights:
            self.model.set_weights(self.init_weights)
            self.target_model.set_weights(self.target_init_weights)
        self.memory.clear()
        self.epsilon = 1.0
        self.epi = 0
