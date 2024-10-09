"""Main DQN agent."""

import torch
import copy
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.utils import (
    get_hard_target_model_updates,
    get_soft_target_model_updates,
)
import tqdm
import numpy as np


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
        preprocessor,
        memory,
        gamma,
        target_update_freq,
        num_burn_in,
        train_freq,
        batch_size,
    ):
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

        self.is_train = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(self, optimizer, loss_func):
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
        self.Q = self.q_network().to(self.device)  # assuming a pytorch model
        self.Q_target = copy.deepcopy(
            self.Q
        )  # Target network is a copy of the Q network
        self.Q_target.requires_grad = False  # No gradient update

        # Define the loss function
        self.loss_func = loss_func  # ofc expect this to be torch.optim.Adam

        # Define the optimizer
        # Defining one for the Q network is sufficient because there is no
        # gradient flow from the target network
        self.optimizer = optimizer(self.Q.parameters(), lr=3e-4)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # Always get the estimated Q values from the target network
        input = self.preprocessor.process_state_for_network(state)
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        return self.Q_target(state).detach()  # just to make sure no gradient flow

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

        # Always get the action from the Q network
        input = self.preprocessor.process_state_for_network(state)
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.Q(input)
            q_values = q_values.cpu().numpy()
        return policy.select_action(q_values, **kwargs)

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
        if len(self.memory) < self.num_burn_in:
            return

        if self.memory.pointer % self.train_freq != 0:
            return

        # Sample a minibatch
        samples = self.memory.sample(self.batch_size)
        processed_samples = self.preprocessor.process_batch(
            samples
        )  # Simple conversion to floats
        processed_samples = [s.as_tuple() for s in processed_samples]
        states, actions, rewards, next_states, dones = tuple(
            np.stack(e) for e in zip(*processed_samples)
        )

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Calculate the target values
        with torch.no_grad():
            next_q_values = self.Q_target(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1).values
            target_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Update your network
        self.optimizer.zero_grad()
        q_values = self.Q(states)  # (B, A)
        selected_q_values = torch.gather(
            q_values, 1, actions.unsqueeze(1)
        ).squeeze()  # (B, 1) -> (B)
        loss = self.loss_func(selected_q_values, target_values)
        loss.backward()
        self.optimizer.step()

        return loss

    def fit(self, env, num_iterations, max_episode_length=None, policy=None):
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
        self.Q_target.train()
        self.is_train = True

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter("runs/experiment_1")

        loss = []
        episode_num = 0
        step = 0
        total_rewards = 0
        done = False
        state = env.reset()
        processed_state = self.preprocessor.process_state_for_memory(state)
        for iter in tqdm.tqdm(range(num_iterations + self.num_burn_in)):
            in_burn_in = iter < self.num_burn_in

            # Take an action according to Q
            action = self.select_action(state, policy=policy, step=not in_burn_in)
            next_state, reward, done = env.step(action)
            total_rewards += reward

            # Do not add s' to the history yet
            processed_next_state = self.preprocessor.process_state_for_memory(
                next_state,
            )
            self.memory.append(
                processed_state, action, reward, processed_next_state, done
            )

            state = next_state

            if not in_burn_in:
                l = self.update_policy()
                if l is not None:
                    loss.append(l.item())

                if iter % self.target_update_freq == 0:
                    self.Q_target = get_hard_target_model_updates(self.Q_target, self.Q)

            step += 1

            if done or (max_episode_length is not None and step >= max_episode_length):
                if not in_burn_in:
                    print(
                        f"Step: {iter-self.num_burn_in}/{num_iterations-self.num_burn_in} Episode {episode_num}: Total reward: {total_rewards} Explore P: {policy.policy.epsilon}"
                    )

                    logs = {
                        "episode_num": episode_num,
                        "total_rewards": total_rewards,
                        "average loss": np.mean(loss),
                    }
                    for key, value in logs.items():
                        writer.add_scalar(key, value, iter)

                step = 0
                total_rewards = 0
                episode_num += 1
                done = False

                # Time to reset
                self.preprocessor.reset()
                state = env.reset()
            writer.close()

    def evaluate(self, env, num_episodes, max_episode_length=None, policy=None):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        self.Q_target.eval()
        self.is_train = False

        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            step = 0
            total_reward = 0
            while not done and (
                max_episode_length is None or step < max_episode_length
            ):
                action = self.select_action(state, policy=policy)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward  # not discounted
                state = next_state
                step += 1
            total_rewards.append(total_reward)

        return total_rewards
