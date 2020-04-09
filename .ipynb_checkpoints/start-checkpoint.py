{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import gym\n",
    "import random\n",
    "import numpy as np\n",
    "from statistics import mean, median\n",
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "LR = 1e-3\n",
    "env = gym.make('CartPole-v0')\n",
    "env.reset()\n",
    "goal_steps=500\n",
    "score_requirement = 50\n",
    "initial_games=10000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def random_games_first():\n",
    "    for episode in range(5):\n",
    "        env.reset()\n",
    "        for t in range(goal_steps):\n",
    "            env.render()\n",
    "            action = env.action_space.sample()\n",
    "            observation, done, reward, info =  env.step(action)\n",
    "            if done:\n",
    "                observation = env.reset()\n",
    "                env.close()\n",
    "                break\n",
    "random_games_first()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ML",
   "language": "python",
   "name": "ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
