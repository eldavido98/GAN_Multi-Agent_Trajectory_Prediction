# Multi-Agent Trajectory Prediction with GAN

For this project I was inspired by the paper [Deep Learning-Based Multimodal Trajectory Prediction with Traffic Light](https://www.mdpi.com/2076-3417/13/22/12339).

## Overview  
This repository contains an implementation of a **multi-agent trajectory prediction** model based on a **Generative Adversarial Network (GAN)**. The goal is to predict future trajectories of multiple agents simultaneously, based on their past positions — producing realistic, socially plausible, and potentially multimodal future paths.

This is particularly useful for simulation, autonomous agents, robotics, crowd forecasting, or game/traffic environments with several interacting agents.


## Features / Key Ideas  
- Uses a GAN architecture (Generator + Discriminator) to learn a distribution over future trajectories, rather than a single deterministic path — enabling **multimodal outputs and diversity**.  
- Supports **multi-agent prediction**: trajectories for *all* agents in a scene are predicted jointly, capturing inter-agent interactions and dependencies.  
- Based on an encoder-decoder architecture with recurrent networks (e.g. LSTM), suitable for sequential trajectory data.  
- Designed to output trajectories in a structured format (e.g. shape `[batch_size, num_agents, num_steps, 2]`, where each step has (x, y) coordinates).  
- Allows integration with a variety of datasets — can be adapted to pedestrian, vehicles, or general multi-agent motion data.


## Repository Structure  
- **data_pre_processing**, **data_processing**, **dataset_definition** : Processe datasets
- **EncDec**, **GAN**, **ResNet** : Model definitions
- **forecasting.py** : Main model
- **train.py** : Training script
- **evaluate.py** : Evaluation / inference script
- **metrics.py** : Metric functions
- **utils** : Helper functions (data loading, preprocessing, plotting, etc.)
- **README.md** : This file


## Usage
Running the code is very simple: you have to download the 'project' folder and run the file 'train.py' if you want to train the model or 'evaluate.py' if you want to evaluate it.


## License & Credits  
Feel free to reuse or adapt this code for educational, research or personal purposes.
