This project includes two folders - Model and Results.

The folder Model includes an agent based model and a reinforcement learning algorithm.

The fodler Results includes the results from three simulations after training.



In Model folder there are 4 files - two are agent based models (MESA_RL_simulation_fisrt.py and MESA_RL_sumilation_final.py).

These two files are identical except of one difference - MESA_RL_sumilation_final.py intialises weights with the learnt weights from previous training. Hence if training model from scratch, fisrt start with MESA_RL_simulation_fisrt.py and then continue with MESA_RL_sumilation_final.py.

DDPG_11.py is an RL algorithm which is used by MESA_RL_simulation scripts. 
