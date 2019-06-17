import torch
import os

def Label(frame,
          track_id,
          type,
          truncated,
          occluded,
          alpha,
          bbox,
          dimensions,
          location,
          rotation_y):
    lbl = {}
    lbl['frame'] = int(frame)
    lbl['track_id'] = int(track_id)
    lbl['type'] = str(type),
    lbl['truncated'] = int(truncated)
    lbl['occluded'] = int(occluded)
    lbl['alpha'] = float(alpha)
    lbl['bbox'] = [float(x) for x in bbox]
    lbl['dimensions'] = [float(x) for x in dimensions]
    lbl['location'] = [float(x) for x in location]
    lbl['rotation_y'] = float(rotation_y)
    return lbl


def get_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    labels = []
    max_frame_idx = -1

    for line in lines:
        l = line.split(' ')
        lbl = Label(l[0],
                    l[1],
                    l[2],
                    l[3],
                    l[4],
                    l[5],
                    l[6:10],
                    l[10:13],
                    l[13:16],
                    l[16])

        if lbl['frame'] > max_frame_idx:
            max_frame_idx = lbl['frame']

        if not lbl['type'][0] == 'DontCare':
            labels.append(lbl)
    ld = {i: [] for i in range(max_frame_idx + 1)}
    {l['frame']: ld[l['frame']].append(l) for l in labels}
    return ld, max_frame_idx


def reshape_labels(labels_dict, input_config):
    """
    Reshaping the labels to a list with information ordered as follows:
    [x_pos, y_pos, length, width, rotation, class]

    :param input_config:
    :return:
    """
    labels = torch.zeros(len(labels_dict), input_config.max_targets_forever, input_config.dim_gt_targets)

    for key, value in labels_dict.items():
        for i, lbl in enumerate(value):
            labels[key, i, 0] = lbl['location'][2]  # (center point) forward, x
            labels[key, i, 1] = -lbl['location'][0]  # (center point) rightward, y
            labels[key, i, 2] = lbl['dimensions'][2]  # length
            labels[key, i, 3] = lbl['dimensions'][1]  # width
            labels[key, i, 4] = lbl['rotation_y']  # rotation (around straight up, z)
            labels[key, i, 5] = 1  # Class TODO: mapping

    return labels


def get_labels_car(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    labels = []
    max_frame_idx = -1
    for line in lines:
        l = line.split(' ')
        lbl = Label(l[0],
                    l[1],
                    l[2],
                    l[3],
                    l[4],
                    l[5],
                    l[6:10],
                    l[10:13],
                    l[13:16],
                    l[16])
        if lbl['frame'] > max_frame_idx:
            max_frame_idx = lbl['frame']

        if lbl['type'][0] == 'Car' or lbl['type'][0] == 'Van':
            labels.append(lbl)

    ld = {i: [] for i in range(max_frame_idx + 1)}
    {l['frame']: ld[l['frame']].append(l) for l in labels}
    return ld, max_frame_idx
