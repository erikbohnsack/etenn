import numpy as np
import math


def coordinate_transform_fafe(state, imu_data, dt):
    """

    :param state:
    :param imu_data:
    :param dt:
    :param object_class:
    :return:
    """
    assert state.shape == (3, 1)
    # Euler integrate angle of ego vehicle
    angle = - imu_data.ru * dt

    # minus angle holds here as well
    rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    translation = np.array([[- imu_data.vl * dt],
                            [imu_data.vf * dt]])

    output_state = np.copy(state)
    output_state[0:2] = rotation_matrix @ state[0:2] - translation
    output_state[2] += angle

    # same argument for minus translation here as above.
    return output_state


def test_coord_transf_fafe():
    angle_to_turn = -math.pi / 4
    dt = 0.1
    ru = angle_to_turn / dt
    vl = 0
    vf = 0
    imu_data = ImuData(ru, vl, vf)

    state = np.array([[10], [10], [math.pi / 4]])
    print("state: {}".format(state))
    output_state = coordinate_transform_fafe(state, imu_data, dt)
    print("output_state: {}".format(output_state))


class ImuData:
    def __init__(self, ru, vl, vf):
        self.ru = ru
        self.vl = vl
        self.vf = vf


def transf(state):
    """
    Returns a transformed state due to KITTI data being different.
    :param state shape (1x6)
    """
    return np.array([[-state[1]], [state[0]], [state[4]]])


class SingleTargetHypothesis:
    """
    Dummy STH class to match with PMBM plotting tools.
    """

    def __init__(self, state):
        self.state = state
        self.variance = None


def translate_center(centers, imu_data, timestep, dt):
    distance_travelled = np.array([[- imu_data.vl * timestep * dt],
                                   [imu_data.vf * timestep * dt],
                                   [0]])

    return centers + distance_travelled
