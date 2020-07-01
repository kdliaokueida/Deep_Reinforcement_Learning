[//]: # (Image References)

[image1]: https://minpy.readthedocs.io/en/latest/_images/pong.gif "Trained Agent"

# Pong Atari

### Introduction

This is an exercise in Deep Reinforcement Learning Nanodgree Program on Udacity.

For this exercise, an agent will be trained to play pong atari.

![Trained Agent][image1]

A reward of +1 is provided for winning the game, and a reward of -1 is provided for losing the game.  

The state are the pixels, and "left fire" and "right fire" were selected out of 6 action states for simplicity.

The task is episodic, and in order to solve the environment.

### Getting Started

1. Follow the setup steps in `pong-REINFORCE.ipynb`

2. Some utility functions are in `pong_utils.py`

3. Environment settings for training multiple agents at the same time are in `parallelEnv.py`
