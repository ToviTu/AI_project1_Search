#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import torch
from torch import nn

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import *

import gymnasium as gym
import ale_py


def create_model(window, input_shape, num_actions, model_name="q_network"):
    """Create the Q-network model.

    You can use any DL library you like, including Tensorflow, Keras or PyTorch.

    If you use Tensorflow or Keras, we highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------

      The Q-model.
    """
    # Using PyTorch here
    # Using this as a class factory

    # this is a CNN model as in the paper
    # not sure how to implement a linear one???
    class DQN(nn.Module):
        def __init__(self):
            super(DQN, self).__init__()
            self.name = model_name
            self.conv1 = nn.Conv2d(window, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, num_actions)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            if len(x.shape) > 3:
                x = x.flatten(
                    start_dim=1, end_dim=-1
                )  # (B, C', H', W') -> (B, C'*H'*W')
            else:
                x = x.flatten()  # (C', H', W') -> (C'*H'*W')
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return DQN


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split("-run")[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + "-run{}".format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description="Run DQN on Atari Breakout")
    parser.add_argument("--env", default="Breakout-v0", help="Atari env name")
    parser.add_argument(
        "-o", "--output", default="atari-v0", help="Directory to save data to"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--wandb", default=False, type=bool, help="Random seed")

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your training method.

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Hyperparameters
    input_shape = (84, 84)
    window = 4
    gamma = 0.99
    max_size = int(1e6)
    batchsize = 32
    target_update_frequency = 10000
    lr = 3e-4

    # Create environment
    gym.register_envs(ale_py)
    env = tfrl.utils.AtariWrapper(gym.make(args.env))

    agent = DQNAgent(
        q_network=create_model(window, input_shape, env.action_space.n),
        policy=tfrl.policy.LinearDecayGreedyEpsilonPolicy(
            tfrl.policy.GreedyEpsilonPolicy, "epsilon", 1.0, 0.1, int(1e6)
        ),
        preprocessor=AtariPreprocessor(input_shape),
        memory=tfrl.core.ReplayMemory(max_size, window),
        gamma=gamma,
        target_update_freq=target_update_frequency,
        num_burn_in=50000,
        train_freq=window,
        batch_size=batchsize,
        use_wandb=args.wandb,
    )
    agent.compile(optimizer=torch.optim.Adam, loss_func=mean_huber_loss, lr=lr)
    agent.fit(env, num_iterations=max_size)
    # agent.evaluate(env, num_episodes=5, policy=tfrl.policy.GreedyPolicy())


if __name__ == "__main__":
    main()
