SHELL := /bin/bash

setup:
	conda env create -f env.yml --force

run:
	source activate ML_Project && python3.6 run_the_model.py
	
DQL:
	source activate ML_Project && python3.6 DQLearning.py

remove:
	conda env remove -n ML_Project
