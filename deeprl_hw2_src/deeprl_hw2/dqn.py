"""Main DQN agent."""

import torch
import copy
from deeprl_hw2.utils import (
    get_hard_target_model_updates,
    get_soft_target_model_updates,
)
import tqdm
import numpy as np
from torch.nn import utils as nn_utils
import matplotlib.pyplot as plt
import time
import wandb


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network:
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(
        self,
        q_network,
        policy,
        preprocessor,
        memory,
        gamma,
        target_update_freq,
        num_burn_in,
        batch_size,
        use_wandb=False,
        eval_freq=int(1e4),
        ddqn=False,
    ):
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.policy = policy
        self.eval_freq = eval_freq
        self.ddqn = ddqn

        self.training_log = {
            "iter": 0,
            "n_updates": 0,
            "n_episodes": 0,
            "eval_rewards": [],
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drl")

    def wandb_log(self, dict):
        if self.use_wandb:
            wandb.log(dict)

    def compile(self, optimizer, loss_func, lr):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.

        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.Q = self.q_network().to(self.device)
        self.Q_target = self.q_network().to(self.device)
        # Copy weights
        self.Q_target = get_hard_target_model_updates(self.Q_target, self.Q)
        self.Q_target.requires_grad = False  # No gradient update

        # Define the loss function
        self.loss_func = loss_func

        # Define the optimizer
        # Defining one for the Q network is sufficient because there is no
        # gradient flow from the target network
        self.optimizer = optimizer(self.Q.parameters(), lr=lr)

    @torch.no_grad()
    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """

        # Just a forward pass for target Q network

        if state.ndim == 3:
            assert state.shape == (
                self.preprocessor.window,
                *self.preprocessor.new_size,
            )
            state = state[np.newaxis, ...]
        else:
            assert state.shape[1:] == (
                self.preprocessor.window,
                *self.preprocessor.new_size,
            )

        state_ = self.preprocessor.process_batch(state)

        input = torch.tensor(state_, dtype=torch.float32).to(self.device)
        return self.Q_target(input).detach().cpu().squeeze()  # (B, A) or (A)

    def select_action(self, state, policy, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        if state.ndim == 3:
            assert state.shape == (
                self.preprocessor.window,
                *self.preprocessor.new_size,
            )
            state = state[np.newaxis, ...]
        else:
            assert state.shape[1:] == (
                self.preprocessor.window_length,
                *self.preprocessor.new_size,
            )

        # Always get the action from the Q network
        state_ = self.preprocessor.process_batch(state)
        state_ = torch.tensor(state_, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.Q(state_).detach().squeeze()  # (B, A) or (A)
            q_values = q_values.cpu().numpy()

        return policy.select_action(
            q_values, agent_step=self.training_log["iter"], **kwargs
        )

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Instead, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        loss = None
        selected_q_values = None

        # Do not update the policy if the memory is not filled
        if len(self.memory) < self.num_burn_in:
            return loss, selected_q_values

        # Sample a minibatch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Preprocess the states
        states = self.preprocessor.process_batch(states)
        next_states = self.preprocessor.process_batch(next_states)
        rewards = self.preprocessor.process_reward(rewards)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Calculate the target values
        with torch.no_grad():
            next_q_values = self.Q_target(next_states)

        # Pick actions
        if self.ddqn:
            # Double DQN
            target_actions = torch.argmax(self.Q(next_states), dim=1)  # (B)
        else:
            target_actions = torch.argmax(next_q_values, dim=1)  # (B)
        max_next_q_values = torch.gather(
            next_q_values, 1, target_actions.unsqueeze(1)
        ).squeeze()
        target_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Update your network
        self.optimizer.zero_grad()
        q_values = self.Q(states)  # (B, A)
        selected_q_values = torch.gather(
            q_values, 1, actions.unsqueeze(1)
        ).squeeze()  # (B, 1) -> (B)
        loss = self.loss_func(target_values, selected_q_values)
        loss.backward()
        # Clip the gradients
        nn_utils.clip_grad_norm_(self.Q.parameters(), 10.0)
        self.optimizer.step()
        self.training_log["n_updates"] += 1

        if self.training_log["iter"] % self.target_update_freq == 0:
            self.Q_target = get_hard_target_model_updates(self.Q_target, self.Q)
            print(f"Updated target network at {self.training_log['iter']}")

        return loss, selected_q_values

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        # Training

        # Log the episode
        episode_reward = 0
        episode_length = 0
        losses = []
        q_values = []
        is_eval = False

        state = copy.deepcopy(env.reset())
        self.preprocessor.reset()
        processed_state = self.preprocessor.process_state_for_memory(state)
        while self.training_log["iter"] < num_iterations:
            # Determine if we are in the burn-in period
            in_burn_in = len(self.memory) < self.num_burn_in

            # Select the action
            if not in_burn_in:
                # Select the action using the policy
                action = self.select_action(processed_state, policy=self.policy)
            else:
                # Random action during burn in
                action = env.action_space.sample()

            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

            processed_next_state = self.preprocessor.process_state_for_memory(
                next_state
            )

            self.memory.append(
                processed_state[-1], action, reward, done
            )  # processed_next_state is not needed

            if not in_burn_in:
                self.training_log["iter"] += 1

            # Update the policy
            loss, q_value = self.update_policy()
            if loss is not None and q_value is not None:
                losses.append(loss.item())
                q_values.append(q_value.detach().flatten().cpu().numpy())

            # Prepare to evaluate after the episode
            if not in_burn_in and self.training_log["iter"] % self.eval_freq == 0:
                is_eval = True

            # Update the state
            processed_state = processed_next_state

            if done or (max_episode_length and episode_length >= max_episode_length):

                if not in_burn_in:
                    log = {
                        "Iteration": self.training_log["iter"],
                        "Episode reward": episode_reward,
                        "Episode length": episode_length,
                        "Loss": np.mean(losses),
                        "Q-values": np.mean(np.concatenate(q_values)),
                        "Epsilon": self.policy.policy.epsilon,
                    }

                    if is_eval:
                        is_eval = False
                        eval_rewards = np.mean(self.evaluate(env, num_episodes=20))
                        log["Eval rewards"] = eval_rewards
                        self.training_log["eval_rewards"].append(eval_rewards)

                    print(
                        f"Iteration: {self.training_log['iter']}"
                        + f" Episode reward: {episode_reward}"
                        + f" Episode length: {episode_length}"
                        + f" Loss: {np.mean(losses):.4f}"
                        + f" Q-values: {np.mean(np.concatenate(q_values)):.4f}"
                        + f" Epsilon: {self.policy.policy.epsilon:.4f}"
                        + f" Current memory size: {len(self.memory)}"
                    )

                    self.wandb_log(log)

                # Reset the environment
                self.preprocessor.reset()
                state = copy.deepcopy(env.reset())
                processed_state = self.preprocessor.process_state_for_memory(state)

                losses = []
                q_values = []
                episode_reward = 0
                episode_length = 0

        return self.training_log

    @torch.no_grad()
    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        total_rewards = []
        # Note that we cannot get recent states from the memory

        for seed in range(num_episodes):
            state = env.reset(seed=seed)
            self.preprocessor.reset()

            processed_state = self.preprocessor.process_state_for_memory(state)
            done = False
            step = 0
            total_reward = 0
            while not done and (
                max_episode_length is None or step < max_episode_length
            ):
                q_value = self.calc_q_values(processed_state).numpy()
                action = np.argmax(q_value)

                next_state, reward, done, _, _ = env.step(action)
                processed_next_state = self.preprocessor.process_state_for_memory(
                    next_state
                )
                total_reward += reward  # not discounted
                processed_state = processed_next_state
                step += 1
            total_rewards.append(total_reward)

        print(f"Average rewards: {np.mean(total_rewards)}")
        print(f"All rewards: {total_rewards}")

        return total_rewards
