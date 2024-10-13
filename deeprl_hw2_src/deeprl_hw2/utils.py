"""Common functions you may find useful in your implementation."""

import numpy as np
import gymnasium as gym
import torch

# import tensorflow as tf


# def get_uninitialized_variables(variables=None):
#     """Return a list of uninitialized tf variables.

#     Parameters
#     ----------
#     variables: tf.Variable, list(tf.Variable), optional
#       Filter variable list to only those that are uninitialized. If no
#       variables are specified the list of all variables in the graph
#       will be used.

#     Returns
#     -------
#     list(tf.Variable)
#       List of uninitialized tf variables.
#     """
#     sess = tf.get_default_session()
#     if variables is None:
#         variables = tf.global_variables()
#     else:
#         variables = list(variables)

#     if len(variables) == 0:
#         return []

#     if semver.match(tf.__version__, "<1.0.0"):
#         init_flag = sess.run(
#             tf.pack([tf.is_variable_initialized(v) for v in variables])
#         )
#     else:
#         init_flag = sess.run(
#             tf.stack([tf.is_variable_initialized(v) for v in variables])
#         )
#     return [v for v, f in zip(variables, init_flag) if not f]


def get_soft_target_model_updates(target, source, tau):
    r"""Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    # Implement in PyTorch

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    return target


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    # Implement in PyTorch

    target.load_state_dict(source.state_dict())

    for target_param, param in zip(target.parameters(), source.parameters()):
        assert torch.equal(target_param.data, param.data)

    return target


import matplotlib.pyplot as plt


# Sometimes SpaceInvaders return more than just a frame
class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AtariWrapper, self).__init__(env)
        self.frame_skip = 4
        self.env = env

        self.frame_buffer = [None, None]

    def reset(self, seed=None):
        state = self.env.reset(seed=seed)

        if isinstance(state, tuple):
            state = state[0]

        assert state.shape == (210, 160, 3)
        return state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        if isinstance(state, tuple):
            state = state[0]

        acc_reward = 0
        done = False

        for idx in range(self.frame_skip):
            next_state, reward, done, truncated, info = self.env.step(action)
            acc_reward += reward

            if idx == self.frame_skip - 2:
                self.frame_buffer[0] = next_state
            elif idx == self.frame_skip - 1:
                self.frame_buffer[1] = next_state

            if done:
                break

        final_state = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        assert state.shape == (210, 160, 3)
        return final_state, acc_reward, done, truncated, info
