import numpy as np
import tensorflow.compat.v1 as tf


def rotation_matrix_z(angle: float):
    """Generates appropriate rotation matrix for z-axis given a rotation angle in degrees.

    Args:
        angle (float): rotation angle [degrees]

    Returns:
        tf.Tensor: Rotation matrix
    """
    angle = angle * np.pi / 180
    return tf.constant(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
