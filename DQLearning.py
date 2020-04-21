#!/usr/bin/env python3


import os
import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout


tf.compat.v1.disable_eager_execution()


# Parameters
AGENT_MEMORY = 20000
UPDATE_AFTER_EPISODES = 5
HIDDEN_LAYERS= [64, 128]
BATCH_SIZE = 32
DISCOUNT = 0.997
LEARNING_RATE = 0.001
LOSS_FUNCTION = 'mse'
AVG_OF_LAST = 100
TASK = 2
OPTIMIZER = Adam(lr=LEARNING_RATE)
RENDER_AFTER_EPISODES = 25
q_new = lambda q_max, reward: (reward + DISCOUNT * q_max)
TOTAL_EPISODES = 1000
MAX_EPSILON = 1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.0005
DROP_PRECENT = 0.25
PLAY = True
PLAY_EPISODES = 100
SHOW = False
PLOT = True
TRAIN = True
SAVE_AT_AVG = 500


# Paths
#ENV_PATH is to be replaced by MODIFIED_ENV_PATH
MODIFIED_ENV_PATH = f"./task{TASK}.py"
ENV_PATH = gym.__file__ + "envs/classic_control/cartpole.py"
ENV_PATH = ENV_PATH.replace('__init__.py', '')
MODEL_SAVE_PATH = "./Models"
PATH1 = MODIFIED_ENV_PATH  # Path to the env for specific task
PATH2 = ENV_PATH  # Path to the original env in the system
SAVE = MODEL_SAVE_PATH  # This is where the model will be saved
os.system('cp '+ PATH1 + ' ' + PATH2)
with open(PATH2, 'r') as f:
    print(f"Task file --> {f.readline()}")

# Defining the plotting function
def plot(scores=[]):
    if PLOT:
        plt.clf()
        avg_scores = [sum(scores[:index+1])/(index+1) for index, 
                      score in enumerate(scores)]
        avg_score = sum(scores)/len(scores)
        x_val = [0,len(scores)-1]
        y_val = [avg_score, avg_score]
        plt.plot(scores,'g-', label='current score')
        plt.plot(x_val, y_val,'r-', label='average score')
        plt.plot(avg_scores,'b-', label='average scores')
        if len(scores) >= AVG_OF_LAST:
            avg_of_last = sum(scores[-1*AVG_OF_LAST:])/AVG_OF_LAST
            y_val = [avg_of_last, avg_of_last]
            plt.plot(x_val, y_val, 'k-', label=f'average scores of last {AVG_OF_LAST}')
        plt.xlabel('Epochs--->')
        plt.ylabel('Score--->')
        plt.legend()
        plt.pause(0.05)


class Agent:

    def __init__(self):
        self.state = env.reset().tolist()
        self.done = False
        self.total_score = 0
        self.score_per_episode = 0
        self.all_scores = [0]
        self.render = True
        self.epsilon = MAX_EPSILON
        self.avg_score = 0
        self.present_q_model = Sequential()
        self.present_q_model.add(Dense(HIDDEN_LAYERS[0],
                                       input_shape=(env.observation_space.low.size,),
                                       activation='relu'))
        self.present_q_model.add(Dropout(DROP_PRECENT))
        for index in range(1, len(HIDDEN_LAYERS)):
            self.present_q_model.add(Dense(HIDDEN_LAYERS[index],activation='relu'))
            self.present_q_model.add(Dropout(DROP_PRECENT))
        self.present_q_model.add(Dense(env.action_space.n,activation='linear'))
        self.present_q_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER,
                                     metrics=['accuracy'])
        
        self.future_q_model = Sequential()
        self.future_q_model.add(Dense(HIDDEN_LAYERS[0],
                                      input_shape=(env.observation_space.low.size,),
                                      activation='relu'))
        self.future_q_model.add(Dropout(DROP_PRECENT))
        for index in range(1, len(HIDDEN_LAYERS)):
            self.future_q_model.add(Dense(HIDDEN_LAYERS[index],activation='relu'))
            self.future_q_model.add(Dropout(DROP_PRECENT))
        self.future_q_model.add(Dense(env.action_space.n,activation='linear'))
        self.future_q_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, 
                                    metrics=['accuracy'])

        self.future_q_model.set_weights(self.present_q_model.get_weights())
        
        self.memory = deque(maxlen=AGENT_MEMORY)
        self.episodes = 0
        print(self.present_q_model.summary())
        print(self.future_q_model.summary())
        
    def train(self):
        if len(self.memory) < BATCH_SIZE :
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        present_q_values = self.present_q_model.predict(np.array([x[1] for x in batch]))
        #future_q2_values = self.present_q_model.predict(np.array([x[3] for x in batch]))
        future_q_values = self.future_q_model.predict(np.array([x[3] for x in batch]))
        for index, slot in enumerate(batch):
            if not slot[4]:
                #action_pred = np.argmax(future_q2_values[index])
                #q_future_max = future_q_values[index][action_pred]
                q_future_max = np.max(future_q_values[index])
                qnew = q_new(q_future_max, slot[2])
            else:
                qnew = slot[2]
            
            present_q_values[index][slot[0]] = qnew
            
        X = np.array([slot[1] for slot in batch])
        Y = present_q_values
        
        history = self.present_q_model.fit(X, Y, batch_size=BATCH_SIZE,
                                           shuffle=False, verbose = 0)
        
        if self.episodes % UPDATE_AFTER_EPISODES == 0:
            self.future_q_model.set_weights(self.present_q_model.get_weights())
                
    def next_action(self):
        q_values = self.present_q_model.predict(np.array([self.state]))
        
        if np.random.random() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, env.action_space.n)
            
        new_state, reward, done, info = env.step(action)
        self.total_score += reward
        self.score_per_episode += reward
        
        if done:
            self.all_scores.append(self.score_per_episode)
            self.score_per_episode = 0
            if PLOT:
                plot(self.all_scores)
            self.episodes +=1
            if self.episodes % RENDER_AFTER_EPISODES == 0:
                self.render = True
            else:
                self.render = False
                
            if self.episodes % AVG_OF_LAST == 0:
                self.avg_score = self.total_score/AVG_OF_LAST
                print(f"Average Score = {self.avg_score}")
                self.total_score = 0
                        
        if self.render:
            #env.render()
            None
            
            
        self.memory.append([action, self.state, reward, new_state, done])
        self.state = new_state.tolist() if not done else env.reset()
        
        self.train()
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)
        
        return done



# Run the games and train the model
if TRAIN:
    env = gym.make('CartPole-v1')
    agent = Agent()
    max_score = 0
    max_avg = 0
    for episode in tqdm(range(TOTAL_EPISODES)):
        while True:
            stop = agent.next_action()
            if stop:
                if max_score <= agent.all_scores[-1]:
                    try:
                        os.system('rm '+ f"{SAVE}/DQL_model-{TASK}-.h5")
                    except:
                        print("No model to remove")
                    try:
                        max_score = agent.all_scores[-1]
                        agent.present_q_model.save(f"{SAVE}/DQL_model-{TASK}-.h5")
                    except KeyboardInterrupt:
                        max_score = agent.all_scores[-1]
                        agent.present_q_model.save(f"{SAVE}/DQL_model-{TASK}-.h5")
                        raise KeyboardInterrupt
                if episode >= AVG_OF_LAST:
                    max_avg = max(sum(agent.all_scores[-1*AVG_OF_LAST:])/AVG_OF_LAST, max_avg)
                    if max_avg >= SAVE_AT_AVG:
                        agent.present_q_model.save(f"{SAVE}/DQL_avg_model-{TASK}-{max_avg}.h5")
                        print("saved")
                break

