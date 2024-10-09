"""Loss functions."""

# import tensorflow as tf
# import semver

import torch


def huber_loss(y_true, y_pred, max_grad=1.0):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The huber loss.
    """
    # Going to implement this in torch

    # Assuming y_true of shape (batch_size)
    diff = y_true - y_pred

    # Assuming max_grad is a scalar
    loss = torch.where(
        torch.abs(diff) < max_grad,
        0.5 * diff**2,
        max_grad * (torch.abs(diff) - 0.5 * max_grad),
    )

    return loss


def mean_huber_loss(y_true, y_pred, max_grad=1.0):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor, torch.Tensor
      Target value.
    y_pred: np.array, tf.Tensor, torch.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor, torch.Tensor
      The mean huber loss.
    """
    batch_loss = huber_loss(y_true, y_pred, max_grad)

    return torch.mean(batch_loss)
