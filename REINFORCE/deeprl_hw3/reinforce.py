import gym
import numpy as np
import tensorflow as tf
from .imitation import test_cloned_policy


def get_total_reward(env, model, training=True):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment.
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    total_reward = 0
    state, _ = env.reset()
    is_done = False
    truncated = False
    while not is_done and not truncated:
        action = np.argmax(model.predict_on_batch(state[np.newaxis, ...])[0])
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward


def choose_action(model, observation, training=True):
    """choose the action

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float
        probability of action 1
    action: int
        the action you choose
    """

    probs = model.predict_on_batch(observation[np.newaxis, ...])[0]

    if training:
        action = np.random.choice(len(probs), p=probs)
    else:
        action = np.argmax(probs)
    p = probs[action]

    return p, action


# for some reason, the following code does not work
class pseudo_loss:

    def call(self, model, observation, action, reward):
        # Compute a pseudo loss for policy gradient
        # Actual gradient computation is done by the optimizer

        # Assume batch observation, action, reward
        probs = model(observation, training=True)
        log_probs = np.log(probs)
        action_log_probs = log_probs[:, action]

        loss = -np.mean(action_log_probs * reward)
        return loss


def reinforce(env, model, optimizer):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    """

    # Implement training algorithm here
    # loss = pseudo_loss()

    @tf.function
    def apply_gradients(gradients):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    batch_observation = []
    batch_action = []
    batch_reward = []

    obs, _ = env.reset()
    done = False
    truncated = False
    eps_len = 0
    eps_reward = 0

    do_update = False
    while not do_update:
        batch_observation.append(obs)

        _, action = choose_action(model, obs)
        batch_action.append(action)
        eps_len += 1

        # Take action
        obs, reward, done, truncated, _ = env.step(action)
        eps_reward += reward

        # If end of episode, append reward
        if done or truncated:
            batch_reward += [eps_reward] * eps_len

            # Reset
            obs, _ = env.reset()
            done = False
            truncated = False
            eps_len = 0
            eps_reward = 0

            do_update = True
            # If enough samples
            # OpenAI's implementation uses batching but the vanila REINFORCE does not
            # if len(batch_observation) >= batch_size:
            #     do_update = True

    # Update model
    batch_observation = np.stack(batch_observation)
    batch_action = np.array(batch_action)
    batch_reward = np.array(batch_reward)
    with tf.GradientTape() as tape:
        # loss = loss.call(model, batch_observation, batch_action, batch_reward)
        action_probs = model(batch_observation, training=True)
        action_masks = tf.one_hot(batch_action, action_probs.shape[1])
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)
        loss = -tf.reduce_sum(log_probs * batch_reward)
    gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    apply_gradients(gradients)
    return


def train(env, model, optimizer, num_step=500):
    mean_rewards = []
    min_rewards = []
    max_rewards = []
    eval_epochs = []
    for i in range(num_step):
        reinforce(env, model, optimizer)

        if i % (num_step // 20) == 0:
            eval_rewards = []
            for _ in range(100):
                eval_rewards.append(get_total_reward(env, model, training=False))
            print(f"Epoch: {i}, Mean Reward: {np.mean(eval_rewards)}")
            # test_cloned_policy(env, model, num_episodes=10, render=False)
            mean_rewards.append(np.mean(eval_rewards))
            min_rewards.append(np.min(eval_rewards))
            max_rewards.append(np.max(eval_rewards))
            eval_epochs.append(i)
    return {
        "mean_rewards": mean_rewards,
        "min_rewards": min_rewards,
        "max_rewards": max_rewards,
        "eval_epochs": eval_epochs,
    }
