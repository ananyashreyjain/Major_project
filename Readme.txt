Steps for setting up the environment:

	1) Enter make setup in terminal to install the environment.
	2) Enter make remove in terminal to uninstall the environment.

Steps for running the trained model:

	1) To run the model enter make run in terminal. It will automatically activate the 
	   new environment if base environment is activate.
	2) Enter the task number that you want to execute. It automatically replace the code in your
	   gym installation, you don't have to do it manually.
	3) Default model is set to Deep Q learning which the model that we are presenting
	   press enter to continue with it, we also tried a non Q based Reinforcement Learning 
	   you change the model to that by entering DL.
	4) Animated graph will popup showing score at every episode.
	5) Green line shows the score at every episode, red line shows the average score and
	   blue line records average score on every episode
	6) After 100 episodes it will show the average score on those 100 episodes

Steps for running the training model:

	1) For DQLearning training code enter make DQL
	2) For Deep Learning training code open jupyter notebook and run DLapproach.ipynb
	3) For Q Learning training code open jupyter notebook and run Qlearning.ipynb
