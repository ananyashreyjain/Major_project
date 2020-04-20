#!/usr/bin/env python

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from statistics import mean, median

# Parameters:
TASK = 3
LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()

# Paths:
#ENV_PATH is to be replaced by MODIFIED_ENV_PATH
MODIFIED_ENV_PATH = f"./task{TASK}.py"
ENV_PATH = gym.__file__ + "envs/classic_control/cartpole.py"
ENV_PATH = ENV_PATH.replace('__init__.py', '')
MODEL_SAVE_PATH = "./models"
PATH1 = MODIFIED_ENV_PATH  # Path to the env for specific task
PATH2 = ENV_PATH  # Path to the original env in the system
SAVE = MODEL_SAVE_PATH  # This is where the model will be saved
os.system('cp '+ PATH1 + ' ' + PATH2)
with open(PATH2, 'r') as f:
    print(f"Task file --> {f.readline()}")


# This function play the game based on the params dictionary passed to it
# If the model param is set then it will use that model to predict, else the action will be selected randomly
def run(params, input=False):
    env.reset()
    for game in range(params["initial_games"]):
        # print(game)
        score = 0
        env.reset()
        game_data = np.array([])
        prev_observation = None
        for step in range(params["goal_steps"]):
            if prev_observation is None or params['model'] is None:
                action = env.action_space.sample()
            else :
                prob_values = model.predict(np.array([prev_observation]))[0]
                for index, prob in enumerate(prob_values):
                    if(prob_values[index]==max(prob_values)):
                        action=index
            observation, reward, done, info =  env.step(action)
            score += reward
            if prev_observation is not None:
                prev_observation=np.concatenate((prev_observation, action), axis=None)
                if(len(game_data)==0):
                    game_data = prev_observation
                else:
                    game_data=np.vstack((game_data, prev_observation))
            if done:
                break
            prev_observation=observation
        if params['verbose']:
            print(score)
        if score>=params['score_requirement']:
            if len(params['training_data'])==0:
                params['training_data']=game_data
                params['score']=np.array([score])
            else:
                params['training_data']=np.concatenate((params['training_data'], game_data))
                params['score']=np.concatenate((params['score'], np.array([score])))
        # print('Average score:',mean(params['score']))
        # print('Median score:',median(params['score']))
        # print(Counter(params['score']))


### TRAINING ###
# Params dict for training
params = dict(
    goal_steps = 500,
    score_requirement = 50,
    initial_games = 25000,
    training_data = np.array([]),
    action = 2,
    score = np.array([]),
    model = None,
    verbose = False
)
print("Generating data")
run(params)
print("Data genration done")
# Create arrays for training
print("Training model")
X=None  # Features: States of the system
Y=None  # Labels: Actions taken for that state
for move in params['training_data']:
    if X is None:
        X=move[0:4]
        Y=move[4]
    else:
        X=np.vstack((X,move[0:4]))
        Y=np.concatenate((Y,move[4]), axis=None)
# Set the model
model = tf.keras.Sequential([tf.keras.layers.Dense(64,input_shape=(4,),activation='relu'),
                             tf.keras.layers.Dense(128,activation='relu'),
                             tf.keras.layers.Dense(2, activation='softmax'),
                                
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# Training:
model.fit(X,Y, epochs=10)
print("Model training done")


#### EVALUATION ###
# Params dict for evaluation
params = dict(
    goal_steps = 500,
    score_requirement = 50,
    initial_games = 100,
    training_data = np.array([]),
    action = 2,
    score = np.array([]),
    model = model,
    verbose = True
)
# Play the game
run(params)

# Print the scores:
print('Average score:',mean(params['score']))
print('Median score:',median(params['score']))
print(Counter(params['score']))

#Save model
params['model'].save(f"{SAVE}/DL_model_task-{TASK}-{mean(params['score'])}.h5")