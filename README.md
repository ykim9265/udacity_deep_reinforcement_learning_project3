
# Overview

This repository contains my "Project 3" submission for Udacity's nanodegree program, 
"Deep Reinforcement Learning" started in late 2020.

It provides an implementation of a learning agent that solves the "Tennis" environment:
[click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

# Project Details

![Tennis Environment](./img/tennis.png "Tennis Environment")


In the "Tennis" environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

    After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


In the "Tennis" environment, an agent is a double-jointed arm that tries to position the aim at the 
goal location as long as possible. 
The state space has 33 dimensions, containing the arm's position, rotation, velocity, and angular velocities.
With this state information, the agent has to maximize total reward by selecting actions for the joints of the arm.
At each time step, the 4 available actions correspond to torques associated with the 2 joints of the agent's arm.
Each torque action is associated with a value in range [-1, 1].

The task is episodic, meaning it has clear beginning and an end, and everything gets reset at the start of each episode.
In terms of rewards, +0.1 is provided each time step when the arm is at the goal location.
The environment is considered "solved" when the agent has +30 average score during past 100 consecutive episodes.


There are two different ways to solve the Reacher environment: 1) single agent vs 2) multiple agents.
In the case of the single agent version, the agent must get an average score of +30 over 100 consecutive episodes.
In the case of the multiple agents version, scores across all agents are averaged, and average of this average score overe 100 consecutive episodes is taken. If the double average score of +30 reached over 100 consecutive episodes, then the environment is solved.

Note that to run this Reacher enviroment, the user has to use the provided Unity environment file, and *not* use the environment on the ML-agents GitHub page.


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



The zip file needs to be placed in the same folder as the training notebook, and uncompressed.

#### 2. Explore the Environment

If the Unity environment has been set up correctly, then the notebook should 
run like the one shown below in the video: [click here](https://youtu.be/kxDvrkg8ep0)

## Instructions

To train an agent for the Tennis environment, you can run the jupyter 
notebook `training_code_tennis.ipynb` in the python environment that you created above.


