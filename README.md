
# Overview

This repository contains my "Project 3" submission for Udacity's nanodegree program, 
"Deep Reinforcement Learning" started in late 2020.

It provides an implementation of a learning agent that solves the "Tennis" environment:
[click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

# Project Details

![Tennis Environment](./img/tennis.png "Tennis Environment")


In the "Tennis" environment, there are 2 agents, each holding a racket. 
For each episode, a tennis ball is dropped onto one of the agents, and each agent has to hit the ball over the net.
As for the reward, if an agent hits the ball over the net, it gets a reward of +0.1.
If the ball hits the ground or out of bounds, then the agent receives a reward of -0.01.
Thus each agent tries to maximize its sum of reward for each episode.

Each agent gets its own state, which contains position and velocities of the ball and racket.
In terms of actions, each agent can 1) move toward or away from the net, or 2) jump.

The task is episodic, meaning it has clear beginning and an end, and everything gets reset at the start of each episode.
Since there are two agents, a maximum score is calculated per episode over the two agents.
And the environment is considered "solved" when average of this maximum score of +0.5 over past 100 consecutive episodes is achieved.

Note that to run this Tennis enviroment, the user has to use the provided Unity environment file, and *not* use the environment on the ML-agents GitHub page.




# Getting Started

## Installation
The following instructions have been tested on the linux ubuntu (18.04) environment.

### Python virtual environment
1. Create a python version 3.6 virtual environment. You can use `conda` or `pyenv`.
2. Once you `activate` into the environment, issue the following statements to install required dependencies:

Here is the command:

    pip install numpy matplotlib jupyter torch unityagents

If using `conda`, one can isssue the following command:

    conda create -n myenv python=3.6 numpy matplotlib jupyter torch
    conda activate myenv
    pip install unityagents

Note that I could not install `unityagents` in python3.7, but was able to in python3.6. Hence python version 3.6 is required.

### Unity Environment

#### 1. Download the Unity Environment

To run the notebook, you need to download one of the prepared Unity environment files matching your setup:


1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the same folder as the training code notebook `training_code_tennis.ipynb`, and unzip (or decompress) the file.


#### 2. Explore the Environment

If the Unity environment has been set up correctly, then the notebook should 
run like the one shown below in the video: [click here](https://youtu.be/kxDvrkg8ep0)

## Instructions

To train an agent for the Tennis environment, you can run the jupyter 
notebook `training_code_tennis.ipynb` in the python environment that you created above.


