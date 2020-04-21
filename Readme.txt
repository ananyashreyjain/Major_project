RUNNING USING CONDA

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

-------------------------------------------------------------------------------------

FOR NON CONDA USERS

1) Run the run_the_model.py script.
2) You will be asked for the task for which the environment is to be loaded.
3) You will then be asked if you want to run the model trained with the DQL method (our final approach which we want to submit) or our modified DL method for reinforcement learning (details are in the report).
   Hit enter to run the DQL method.
   Type DL and hit enter for the other method.
3) Once you enter the task, the approapriate environment will be set and the appropriate model will be loaded from the Models folder.
4) A graph for the score of each run will be plotted and at the end the average score will be printed.