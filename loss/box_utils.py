# -*- coding: utf-8 -*-
import torch


def point_form_fafe(boxes):
    """
    More general version of below point form conversion function.

    :param boxes: Input any number of dimensions, but make sure that the last dimensions
    is of the following type [x, y, l, w].
    :return: Returns same number of dimensions, but instead of center-sized bboxes, it returns corner
    coordinate (x1y1x2y2) bboxes.
    """

    num_dim = len(boxes.shape) - 1
    xmin = boxes[..., 0] - boxes[..., 2] / 2
    xmax = boxes[..., 0] + boxes[..., 2] / 2

    ymin = boxes[..., 1] - boxes[..., 3] / 2
    ymax = boxes[..., 1] + boxes[..., 3] / 2
    return torch.cat(
        (xmin.unsqueeze(num_dim), ymin.unsqueeze(num_dim), xmax.unsqueeze(num_dim), ymax.unsqueeze(num_dim)), num_dim)


def rotation_matrix(angle, device):
    """

    :param angle:
    :return: 3D Rotation matrix around Z axis
    """
    return torch.Tensor([[torch.sin(- angle), torch.cos(- angle), 0],
                         [torch.cos(- angle), - torch.sin(- angle), 0],
                         [0, 0, 1]]).to(device)


def translate_center(centers, imu_data, timestep, dt, device):
    distance_travelled = torch.Tensor([imu_data.vf * timestep * dt,
                                       - imu_data.vl * timestep * dt, 0]).to(device)
    new_centers = centers + distance_travelled.unsqueeze(0).expand(centers.shape[0], -1)
    return new_centers


def rotate_point_3d(point, rotation_matrix, center):
    """

    :param point: 3x1
    :param rotation_matrix: 3x3
    :param center: 3x1
    :return: 3x1
    """
    # Move to origo
    point_c = point - center
    # Rotate and move back
    return torch.mm(rotation_matrix, point_c) + center


def rotate_3d_bbx(boxes, angles, centers, device):
    """

    :param boxes: n x 8 x 3
    :param angle: n x 1
    :return: n x 8 x 3
    """
    n = boxes.shape[0]
    out = torch.empty_like(boxes)
    for box_index in range(n):
        center = centers[box_index].unsqueeze(1)
        rot = rotation_matrix(angle=-angles[box_index], device=device)
        p1 = rotate_point_3d(boxes[box_index][0, :].unsqueeze(1), rot, center).permute(1, 0)
        p2 = rotate_point_3d(boxes[box_index][1, :].unsqueeze(1), rot, center).permute(1, 0)
        p3 = rotate_point_3d(boxes[box_index][2, :].unsqueeze(1), rot, center).permute(1, 0)
        p4 = rotate_point_3d(boxes[box_index][3, :].unsqueeze(1), rot, center).permute(1, 0)
        p5 = rotate_point_3d(boxes[box_index][4, :].unsqueeze(1), rot, center).permute(1, 0)
        p6 = rotate_point_3d(boxes[box_index][5, :].unsqueeze(1), rot, center).permute(1, 0)
        p7 = rotate_point_3d(boxes[box_index][6, :].unsqueeze(1), rot, center).permute(1, 0)
        p8 = rotate_point_3d(boxes[box_index][7, :].unsqueeze(1), rot, center).permute(1, 0)
        out[box_index] = torch.cat((p1, p2, p3, p4, p5, p6, p7, p8), 0)
    return out


def point_form_3d(boxes, z_center, z_height, device):
    """

    :param boxes: x1y1x1x2 form (n x 4)

    :param z_center: Scalar, centerpoint of bounding box in z-axis
    :param z_height: Scalar, height of bbox
    :return: 3d boxes [[x1y1z1],   (n x 8 x 3)
                        ...]
    """
    xmin = boxes[:, 0].unsqueeze(1)
    xmax = boxes[:, 2].unsqueeze(1)

    ymin = boxes[:, 1].unsqueeze(1)
    ymax = boxes[:, 3].unsqueeze(1)
    zmin = torch.Tensor(()).new_full(size=(len(boxes[:, 0]), 1), fill_value=z_center - z_height / 2).to(device)
    zmax = torch.Tensor(()).new_full(size=(len(boxes[:, 0]), 1), fill_value=z_center + z_height / 2).to(device)
    dim = 1
    p1 = torch.cat((xmin, ymin, zmin), dim=dim).unsqueeze(1)
    p2 = torch.cat((xmax, ymin, zmin), dim=dim).unsqueeze(1)
    p3 = torch.cat((xmax, ymax, zmin), dim=dim).unsqueeze(1)
    p4 = torch.cat((xmin, ymax, zmin), dim=dim).unsqueeze(1)
    p5 = torch.cat((xmin, ymin, zmax), dim=dim).unsqueeze(1)
    p6 = torch.cat((xmax, ymin, zmax), dim=dim).unsqueeze(1)
    p7 = torch.cat((xmax, ymax, zmax), dim=dim).unsqueeze(1)
    p8 = torch.cat((xmin, ymax, zmax), dim=dim).unsqueeze(1)
    out = torch.cat((p1, p2, p3, p4, p5, p6, p7, p8), 1)
    return out


def center_size_3d(reg_targets, z_center, device):
    """
    Takes x and y-columns of reg_targets tensor ands adds a z-column with the value specified
    :param reg_targets: Regression targets tensor, Shape (n x reg_targets_dim) where n is how many positive points
    :param z_center: Scalar
    :param device: The device to use (cpu or cuda:n)
    :return: (n x 3)
    """
    return torch.cat((reg_targets[:, 0].unsqueeze(1),
                      reg_targets[:, 1].unsqueeze(1),
                      torch.Tensor(()).new_full(size=(len(reg_targets[..., 0]), 1), fill_value=z_center).to(device)),
                     dim=1)


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def our_intersect(ground_truth, prior):
    """

    :param truths: Ground truth in point form, shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :param priors: Prior boxes in point form. shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :return:
    """
    max_xy = torch.min(ground_truth[..., 2:], prior[..., 2:])
    min_xy = torch.max(ground_truth[..., :2], prior[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[..., 0].mul(inter[..., 1])
    return intersection


def our_jaccard(ground_truth, prior):
    """

    :param truths: Ground truth in point form, shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :param priors: Prior boxes in point form. shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :return: Overlapping sizes for each pair of reshaped truth and prior. shape (nA, nH, nW, #ground_truth)
    """
    inter = our_intersect(ground_truth, prior)
    area_a = (ground_truth[..., 2] - ground_truth[..., 0]) * (ground_truth[..., 3] - ground_truth[..., 1])
    area_b = (prior[..., 2] - prior[..., 0]) * (prior[..., 3] - prior[..., 1])
    union = area_a + area_b - inter
    return inter / (union + 1e-16)


def our_match(truths_pf, priors, lower_threshold=0.45, higher_threshold=0.65):  # , threshold, detection, truth_labels):
    """
    Matches ground truth and prior boxes, returning a mask indicating which prior boxes are connected to ground truth.

    :param truths: Ground truth in point form, shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :param priors: Prior boxes in point form. shape (nA, nH, nW, #ground_truth, #dim_point_form)
    :param threshold: IoU threshold to decide if match or not.

    :return:
    - mask  shape: (nA, nH, nW), bool marking which anchor boxes are "positive samples"
    - gt_mask, shape: (# mask.sum()). A 1D tensor of ground truth indices, marking which gt is connected to each
    positive sample in mask.
    """

    overlaps = our_jaccard(truths_pf, priors)
    #print("Overlaps.size() {}".format(overlaps.shape))

    # ----------------------------------------------------------
    # For all anchor boxes/priors, match ground truth boxes with IoU higher than threshold
    # ----------------------------------------------------------
    best_gt_overlap, best_gt_index = overlaps.max(3, keepdim=False)

    # ----------------------------------------------------------
    # For each ground truth, find best corresponding anchor/prior
    # When index found, bump the value of that in the
    # ----------------------------------------------------------
    best_prior_overlap, best_prior_index = overlaps.max(0, keepdim=False)

    # Find max value along height dimension.
    max_prior_height_overlap, max_prior_height_index = best_prior_overlap.max(0, keepdim=False)

    # Find max value along the width dimension, of the already maxed height dimension.
    max_prior_width_overlap, max_prior_width_index = max_prior_height_overlap.max(0, keepdim=False)

    _max_height = max_prior_height_index.permute(1, 0)[:, max_prior_width_index]
    for gt_index in range(len(_max_height)):
        w_index = int(max_prior_width_index[gt_index])
        h_index = int(_max_height[gt_index][gt_index])
        anchor_index = int(best_prior_index[h_index][w_index][gt_index])
        best_gt_overlap[anchor_index][h_index][w_index] = 2
        best_gt_index[anchor_index][h_index][w_index] = gt_index

    # ----------------------------------------------------------
    # Check IoU vs threshold
    # ----------------------------------------------------------
    mask = best_gt_overlap > higher_threshold
    neg_mask = best_gt_overlap < lower_threshold

    _, gt_mask = overlaps[mask].max(1)

    # print("\tOverlaps: {}".format(overlaps.shape))
    # print('\tMask: {}'.format(mask.shape))
    # print('\tGT Mask: {}'.format(gt_mask))

    return mask, gt_mask, neg_mask

    # ----------------------------------------------------------
    # Check labels
    # ----------------------------------------------------------
    # print("\tPrior Overlap: {}\n\tPrior Index: {}".format(best_prior_overlap.shape, best_prior_index.shape))
    # print("\tGT Overlap: {}\n\tGT Index: {}".format(best_gt_overlap.shape, best_gt_index.shape))
    # TODO: MARK THEM APPROPRIATELY IN MASKS

    pass


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
