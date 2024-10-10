"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
import copy

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor

FRAME_SHAPE = (84, 84)


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        # assuming storing raw frames for now
        self.history = [
            np.zeros(FRAME_SHAPE, dtype=np.uint8) for _ in range(self.history_length)
        ]

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""

        # Take maximum of the last 2 frames

        # Add to history
        self.history.append(state)

        if len(self.history) > self.history_length:
            self.history.pop(0)

        # if use AtariPreprocessor and then HistoryPreprocessor
        # expect the output shape to be (m, 84, 84)
        return np.stack(self.history, axis=0)

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history = [
            np.zeros(FRAME_SHAPE, dtype=np.uint8) for _ in range(self.history_length)
        ]

    def get_config(self):
        return {"history_length": self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size
        self.past_frames = [np.zeros(new_size, dtype=np.uint8) for _ in range(4)]

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        # assuming state is an image (210, 160, 3)
        # state = copy.deepcopy(state)
        # Let us process with Image module
        img = Image.fromarray(state)

        # rescale the image
        img = img.resize(self.new_size)

        # convert to greyscale
        img = img.convert("L")

        # convert to numpy array
        processed_state = np.array(img, dtype=np.uint8)

        # max over the last two frames
        max_processed_state = np.maximum.reduce(
            self.past_frames + [processed_state], axis=0
        )

        # update the past frames
        self.past_frames.append(processed_state)
        self.past_frames.pop(0)

        return max_processed_state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        # assuming state is an image (210, 160, 3)
        # state = copy.deepcopy(state)
        # Let us process with Image module
        img = Image.fromarray(state)

        # rescale the image
        img = img.resize(self.new_size)

        # convert to greyscale
        img = img.convert("L")

        # convert to numpy array
        processed_state = np.array(img, dtype=np.float32) / 255.0
        return processed_state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        # assume samples are preprocessed already
        return samples.astype(np.float32) / 255.0

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1, 1)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """

    def __init__(self, input_shape, window):

        self.history = HistoryPreprocessor(window)
        self.atari = AtariPreprocessor(input_shape)

    def process_state_for_network(self, state):
        """Process the state for the network."""
        state = self.atari.process_state_for_network(state)
        state = self.history.process_state_for_network(state)
        return state

    def process_state_for_memory(self, state):
        """Process the state for the memory."""
        state = self.atari.process_state_for_memory(state)
        state = self.history.process_state_for_memory(state)
        return state

    def process_batch(self, samples):
        """Process the batch for the network."""
        samples = self.atari.process_batch(samples)
        samples = self.history.process_batch(samples)
        return samples

    def reset(self):
        """Reset the history."""
        self.history.reset()
        self.atari.reset()
