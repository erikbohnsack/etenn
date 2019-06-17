import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import os
import torch
from matplotlib.lines import Line2D


def plot_BEV(grid):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.gca(projection='3d')
    ax.voxels(grid)
    ax.view_init(azim=0, elev=90)
    plt.show()


def plot_lidar(path, sequence, frame, split='training'):
    data_path = os.path.join(path, split, 'velodyne', str(sequence).zfill(4))

    file = os.path.join(data_path, str(frame).zfill(6) + '.bin')
    pc = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
    return draw_lidar(pc)


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None,
               off_screen_rendering=True):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    mlab.options.offscreen = off_screen_rendering
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    # TOP_Y_MIN = -20
    # TOP_Y_MAX = 20
    # TOP_X_MIN = 0
    # TOP_X_MAX = 40
    # TOP_Z_MIN = -2.0
    # TOP_Z_MAX = 0.4

    # x1 = TOP_X_MIN
    # x2 = TOP_X_MAX
    # y1 = TOP_Y_MIN
    # y2 = TOP_Y_MAX
    # mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)

    return fig


def draw_points_3d(points, fig, color=None, pts_scale=0.2, pts_mode='sphere'):
    # Must use CPU due to mlab....
    if points.is_cuda:
        points = points.to(torch.device("cpu"))

    if points.requires_grad:
        points = points.detach()

    # For loop as MayaVI otherwise decides point size based on distance in between the points. Not what we want
    for i in range(len(points[:, 0])):
        mlab.points3d(points[i, 0], points[i, 1], points[i, 2], color=color, mode=pts_mode, colormap='gnuplot',
                      scale_factor=pts_scale, figure=fig)
    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None, probabilities=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    # Must use cpu here due to mlab
    if gt_boxes3d.is_cuda:
        gt_boxes3d = gt_boxes3d.to(torch.device("cpu"))

    for n in range(num):
        b = gt_boxes3d[n]

        if b.requires_grad:
            b = b.detach()

        if color_list is not None:
            color = color_list[n]
        if draw_text:
            if probabilities is None:
                mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
            else:
                _txt = str(n) + ' [' + str(round(probabilities[n].item(), 2)) + ']'
                mlab.text3d(b[4, 0], b[4, 1], b[4, 2], _txt, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_gospa(gospa_score, fig, time_index, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1)):
    _txt = 'Gospa t+' + str(time_index) + ': [' + str(round(gospa_score, 2)) + ']'
    mlab.text3d(50 - 2 * time_index, -22 - 0.25 * time_index, 9 - 1.2 * time_index, _txt, scale=text_scale, color=color,
                figure=fig)
    return fig


def plot_grad_flow(named_parameters, filename):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(filename)
    plt.close()