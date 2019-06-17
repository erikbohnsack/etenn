import torch
from numba import jit
import numpy as np
from cfg.config import InputConfig
from math import atan2, sqrt
from scipy.spatial.distance import cdist


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


@jit(nopython=True, parallel=True)
def format_to_BEV_jit(bev_map: np.array, points: np.array, grid_limits: np.array, grid_sizes: np.array):
    """

    :param bev_map:
    :param points:
    :param grid_limits: np.array([[xmin, xmax],    Shape [3, 2]
                                  [ymin, ymax],
                                  [zmin, zmax]])
    :param grid_sizes: np.array([grid_size_x, grid_size_y, grid_size_z]

    :return:
    """
    for i in range(points.shape[0]):
        if grid_limits[0, 0] < points[i, 0] < grid_limits[0, 1] and \
                grid_limits[1, 0] < points[i, 1] < grid_limits[1, 1] and \
                grid_limits[2, 0] < points[i, 2] < grid_limits[2, 1]:
            pixel_x = np.int(np.floor((points[i, 0] - grid_limits[0, 0]) / grid_sizes[0]))
            pixel_y = np.int(np.floor((points[i, 1] - grid_limits[1, 0]) / grid_sizes[1]))
            pixel_z = np.int(np.floor((points[i, 2] - grid_limits[2, 0]) / grid_sizes[2]))
        bev_map[pixel_x][pixel_y][pixel_z] = 1


def format_to_BEV(bev_map: torch.Tensor, points: np.array, config: InputConfig):
    """

    :param bev_map:
    :param points:
    :param config:
    :return:
    """
    number_of_points = points.shape[0]

    for i in range(number_of_points):
        if config.x_min < points[i, 0] < config.x_max and \
                config.y_min < points[i, 1] < config.y_max and \
                config.z_min < points[i, 2] < config.z_max:
            pixel_x = np.int(np.floor((points[i, 0] - config.x_min) / config.x_grid_size))
            pixel_y = np.int(np.floor((points[i, 1] - config.y_min) / config.y_grid_size))
            pixel_z = np.int(np.floor((points[i, 2] - config.z_min) / config.z_grid_size))

            bev_map[pixel_x][pixel_y][pixel_z] = 1


def within_fov(point, min_angle=0.78, max_angle=2.45, max_radius=100):
    angle = atan2(point[0], point[1])
    radius = sqrt(point[0] ** 2 + point[1] ** 2)
    return min_angle < angle < max_angle and radius < max_radius


def too_close(inference_reg, distance_threshold):
    meas = []
    for ir in inference_reg:
        a = ir[0]
        b = ir[1]
        meas.append([a, b])
    meas = np.array(meas)

    if len(meas) == 0:
        return inference_reg

    dist = cdist(meas, meas, metric='euclidean')
    dist_bool = (dist < distance_threshold)
    nonzero_bool = dist != 0.

    too_close_bool = np.multiply(dist_bool, nonzero_bool)

    # Check matching boxes
    avoid = []  # If data point already taken by matched point, ignore
    jointly_reasoned = []  # Store jointly_reasoned data.

    # Loop over rows in matching matrix
    for i in range(too_close_bool.shape[0]):

        if i in avoid:
            continue

        if not too_close_bool[i].all():
            match = [i] + np.argwhere(too_close_bool[i]).reshape(-1, ).tolist()
            avoid.extend(match)

            max_prob = 0
            max_ix = None
            for ix in match:
                if inference_reg[ix][-1] > max_prob:
                    max_prob = inference_reg[ix][-1]
                    max_ix = ix
            jointly_reasoned.append(inference_reg[max_ix].unsqueeze(0))
        else:
            jointly_reasoned.append(inference_reg[i])

    return jointly_reasoned
