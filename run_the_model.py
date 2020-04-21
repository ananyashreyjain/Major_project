#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
from tqdm import tqdm
import random
import os
import gym
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

MODEL_SCORE = input("Score of the input model ")

TASK = 1
RENDER_AFTER_EPISODES = 25
PLAY_EPISODES = 100
SHOW = False  # For rendering
PLOT = True  # For plotting
AVG_OF_LAST = 100

#User Settings
#ENV_PATH is to be replaced by MODIFIED_ENV_PATH
APPROACH = "DQL"  # The approach used to train this model
MODIFIED_ENV_PATH = f"./task{TASK}.py"
ENV_PATH = gym.__file__ + "envs/classic_control/cartpole.py"
ENV_PATH = ENV_PATH.replace("__init__.py", "")
print(ENV_PATH)
MODEL_PATH = f"./Models/{APPROACH}_model_task-{TASK}-{MODEL_SCORE}.h5"
PATH1 = MODIFIED_ENV_PATH
PATH2 = ENV_PATH
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
        plt.pause(0.01)


# Running the model
env = gym.make('CartPole-v1')
trained_agent = load_model(MODEL_PATH)
state = env.reset()
scores = []
for _ in tqdm(range(PLAY_EPISODES)):
    env.reset()
    score_per_episode = 0
    while True:
        action = np.argmax(trained_agent.predict(np.array([state])))
        state, reward, done, info = env.step(action)
        score_per_episode += reward
        if SHOW:
            env.render()
        if done:
            scores.append(score_per_episode)
            break
    if PLOT:
        plot(scores)
plt.pause(5)
print(sum(scores)/PLAY_EPISODES)