"""Core classes."""

import numpy as np
import matplotlib.pyplot as plt
import time


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """

    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

        assert state.ndim == 2 and state.ndim == 2
        assert state.dtype == np.uint8 and next_state.dtype == np.uint8

    def as_tuple(self):
        return (self.state, self.action, self.reward, self.next_state, self.is_terminal)


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """

    def __init__(self, max_size, window_length, state_shape):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size
        self.window_length = window_length
        self.state_shape = state_shape

        # Implement a ring buffer but differentiate each elements
        # easier to retrieve without for loops
        # store state and next_state together since we know their order
        # Store individual frames
        self.frames = np.zeros((max_size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((max_size,), dtype=np.int32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def append(self, state, action, reward, done):
        """
        Add a sample to the replay memory.
        """

        # Check not normalized
        assert np.max(state) > 1
        assert state.shape == self.state_shape
        assert state.dtype == np.uint8
        assert state.ndim == 2  # must not be stacked

        # Insert and update the pointer
        self.frames[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.max_size
        self.size = max(self.size, self.position)

    # Not planning to use this
    def end_episode(self, final_state, is_terminal):
        """
        Set the final state of an episode and mark whether it was a true terminal state.
        """
        if self.size == 0:
            return
        last_idx = (self.position - 1) % self.size
        self.dones[last_idx] = is_terminal

    def _get_stacked_frames(self, idx):
        """
        Retrieve stacked frames for a given index.
        Stack up to window_length frames, handling episode boundaries.
        """
        end_idx = idx + 1  # The index is inclusive for the latest frame
        start_idx = end_idx - self.window_length  # Window of frames

        indices = np.arange(start_idx, end_idx) % self.size
        candidates = self.frames[indices]

        valid = self.dones[indices[:-1]] == False  # Ignore current frame

        if valid.all():
            frames_stack = candidates
        else:
            padding_size = np.where(~valid)[0][-1] + 1
            padding = np.zeros_like(candidates[0])[np.newaxis, ...]
            padding = np.repeat(padding, padding_size, axis=0)
            frames_stack = np.concatenate([padding, candidates[padding_size:]], axis=0)

        assert frames_stack.shape == (self.window_length, *self.state_shape)

        return frames_stack

    # Retrieve samples
    def sample(self, batch_size):
        """
        Return a batch of stacked states, actions, rewards, next_states, and dones.
        """
        assert self.size >= batch_size, "Not enough samples in memory!"

        indices = np.random.randint(0, self.size - 1, size=batch_size)

        state_batch = np.array(
            [self._get_stacked_frames(idx) for idx in indices], dtype=np.uint8
        )
        next_state_batch = np.array(
            [self._get_stacked_frames((idx + 1) % self.max_size) for idx in indices],
            dtype=np.uint8,
        )
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        done_batch = self.dones[indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def clear(self):
        """
        Reset the memory. Deletes all references to the samples.
        """
        # Reinitialize
        self.frames = np.zeros((self.max_size, *self.state_shape), dtype=np.uint8)
        self.actions = np.zeros((self.max_size,), dtype=np.int32)
        self.rewards = np.zeros((self.max_size,), dtype=np.float32)
        self.dones = np.zeros((self.max_size,), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.frames[idx],
            self.actions[idx],
            self.rewards[idx],
            self.frames[(idx + 1) % self.max_size],
            self.dones[idx],
        )

    def __iter__(self):
        for i in range(self.size):
            yield self[i]
