"""Functions for imitation learning."""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from keras.models import model_from_json
import numpy as np
import time


def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, "r") as f:
        model = model_from_json(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model


def generate_expert_training_data(expert, env, num_episodes=100, render=True):
    """Generate training dataset.

    Parameters
    ----------
    expert: keras.models.Model
      Model with expert weights.
    env: gym.core.Env
      The gym environment associated with this expert.
    num_episodes: int, optional
      How many expert episodes should be run.
    render: bool, optional
      If present, render the environment, and put a slight pause after
      each action.

    Returns
    -------
    expert_dataset: ndarray(states), ndarray(actions)
      Returns two lists. The first contains all of the states. The
      second contains a one-hot encoding of all of the actions chosen
      by the expert for those states.
    """
    states = None
    actions = None

    ############ TODO #############

    states = []
    actions = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        is_done = False
        truncated = False
        while not is_done and not truncated:
            action = np.argmax(expert.predict_on_batch(state[np.newaxis, ...])[0])
            next_state, _, is_done, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            state = next_state
            if render:
                env.render()
                time.sleep(0.1)
    states = np.stack(states)
    actions = np.eye(env.action_space.n)[actions]
    ###############################

    return states, actions


def test_cloned_policy(env, cloned_policy, num_episodes=50, render=True):
    """Run cloned policy and collect statistics on performance.

    Will print the rewards for each episode and the mean/std of all
    the episode rewards.

    Parameters
    ----------
    env: gym.core.Env
      The CartPole-v0 instance.
    cloned_policy: keras.models.Model
      The model to run on the environment.
    num_episodes: int, optional
      Number of test episodes to average over.
    render: bool, optional
      If true, render the test episodes. This will add a small delay
      after each action.
    """
    total_rewards = []

    for i in range(num_episodes):
        print("Starting episode {}".format(i))
        total_reward = 0
        state, _ = env.reset()
        if render:
            env.render()
            time.sleep(0.1)
        is_done = False
        truncated = False
        while not is_done and not truncated:
            action = np.argmax(
                cloned_policy.predict_on_batch(state[np.newaxis, ...])[0]
            )
            state, reward, is_done, truncated, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(0.1)
        print("Total reward: {}".format(total_reward))
        total_rewards.append(total_reward)

    print(
        "Average total reward: {} (std: {})".format(
            np.mean(total_rewards), np.std(total_rewards)
        )
    )


def wrap_cartpole(env):
    """Start CartPole-v0 in a hard to recover state.

    The basic CartPole-v0 starts in easy to recover states. This means
    that the cloned model actually can execute perfectly. To see that
    the expert policy is actually better than the cloned policy, this
    function returns a modified CartPole-v0 environment. The
    environment will start closer to a failure state.

    You should see that the expert policy performs better on average
    (and with less variance) than the cloned model.

    Parameters
    ----------
    env: gym.core.Env
      The environment to modify.

    Returns
    -------
    gym.core.Env
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_reset = unwrapped_env.reset

    def harder_reset():
        unwrapped_env.orig_reset()
        unwrapped_env.state[0] = np.random.choice([-1.5, 1.5])
        unwrapped_env.state[1] = np.random.choice([-2.0, 2.0])
        unwrapped_env.state[2] = np.random.choice([-0.17, 0.17])
        return unwrapped_env.state.copy(), {}

    unwrapped_env.reset = harder_reset

    return env
