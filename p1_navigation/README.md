[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This is a project in Deep Reinforcement Learning Nanodgree Program on Udacity. 

For this project, an agent will be trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get the workspace ready for training the agent.

### Three DQNs

1. Double DQN (`dqn_agent.py`,`DQNmodel.py`,`model.pth`):
- The model uses 3 fully-connected layers as hidden layers, and each layer has 128 units. 
- Discount rate ($\gamma$) = 0.99
- Stochastic rando ($\epsilon$): initial = 1.0; final = 0.01; decay rate = 0.995
- Learning rate = 0.0005
- Training max episode = 2500

- **Result: 14.99 test on average of 100 episodes**

2. Duel DQN (`duel_dqn_agent.py`,`DuelDQNmodel.py`,`DuelDQN_128x2FC_128FC.pth`):
- The model uses 2 fully-connected 128-unit layers as feature extraction layers, an 128-unit layer as value layer, and an 128-unit layer as advantage layer. 
- Discount rate ($\gamma$) = 0.99
-  Stochastic rando ($\epsilon$): initial = 1.0; final = 0.01; decay rate = 0.995
- Learning rate = 0.0005
- Training max episode = 2500

- **Result: 9.86 test on average of 100 episodes**

3. Prioritized Experience Replay DQN (`replay_dqn_agent.py`,`DQNmodel.py`,`PERDQN_128x3FC.pth`):
- The model is the same as the Double DQN model.
- Discount rate ($\gamma$) = 0.99
- Stochastic random ($\epsilon$): initial = 1.0; final = 0.01; decay rate = 0.995
- Learning rate = 0.0001
- Sample bias ($\alpha$) = 0.6
- Compensate weight ($\beta$): initial = 0.6; final = 1.0; time to saturate = 2000
- Training max episode = 20000
- Replay module is from [click here](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)

- **Result: 8.1 test on average of 100 episodes**