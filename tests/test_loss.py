import unittest
import torch
from loss.box_utils import point_form_fafe, our_intersect, our_jaccard, our_match, center_size_3d


class TestLoss(unittest.TestCase):
    _fov_height = 70
    _fov_width = 80

    def test_our_match(self):
        FloatTensor = torch.FloatTensor
        nB = 5
        nT = 5
        nA = 3
        nH = 44
        nW = 50

        # Could be [nB=1, nT=1, nA=5, nH=44, nW=50, (x1, y1, x2, y2)]
        priors = torch.zeros(1, 1, nA, nH, nW, 4)

        grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)

        priors[..., 0] = grid_x.expand(-1, -1, nA, -1, -1).contiguous()
        priors[..., 1] = grid_y.expand(-1, -1, nA, -1, -1).contiguous()

        anchor_w = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1).mul(2)
        anchor_l = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1)

        priors[..., 2] = anchor_w
        priors[..., 3] = anchor_l

        new_priors = point_form_fafe(priors)
        print(new_priors.shape)
        print(priors[0][0][:][0][0])
        print(new_priors[0][0][:][0][0])

        # ground_truth: (#batches x #conseq_frames x # max targets forever x # ground truth dimensions)

        non_zero_gt = torch.Tensor([[14.9328, 5.4076, 4.4339, 1.8233, -2.1514, 1.0000],
                                    [5.9154, -1.9502, 1.7852, 0.8246, -1.7485, 1.0000],
                                    [39.5655, 8.8195, 5.5303, 1.8953, 1.9189, 1.0000]])
        point_from_gt = point_form_fafe(non_zero_gt[..., 0:4])
        print(point_from_gt.shape)
        print(point_from_gt)

        gt = point_from_gt.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(nA, nH, nW, -1, -1)
        prior = new_priors[0][0][...].unsqueeze(3).expand_as(gt)

        mask, gt_mask = our_match(truths_pf=gt, priors=prior, threshold=0.5)
        print("mask shape: {}".format(mask.shape))
        print("gt mask shape: {}".format(gt_mask.shape))
        print("gt mask: {}".format(gt_mask))

    def test_expand(self):
        FloatTensor = torch.FloatTensor
        nB = 5
        nT = 5
        nA = 5
        nH = 44
        nW = 50

        # Could be [nB=1, nT=1, nA=5, nH=44, nW=50, (x1, y1, x2, y2)]
        priors = torch.zeros(1, 1, nA, nH, nW, 4)

        grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)

        priors[..., 0] = grid_x.expand(-1, -1, nA, -1, -1).contiguous()
        priors[..., 1] = grid_y.expand(-1, -1, nA, -1, -1).contiguous()

        assert grid_x.sum() * nA == priors[..., 0].sum()
        assert grid_y.sum() * nA == priors[..., 1].sum()

        anchor_w = torch.arange(0, float(nA)).view(1, 1, nA, 1, 1)
        anchor_l = torch.arange(0, float(nA)).view(1, 1, nA, 1, 1).mul(2)

        priors[..., 2] = anchor_w
        priors[..., 3] = anchor_l

        assert torch.eq(priors[0][0][0][0][0][2], 0)
        assert torch.eq(priors[0][0][0][0][0][3], 0)
        assert priors[0][0][-1][0][0][2] == nA - 1
        assert priors[0][0][-1][0][0][3] == (nA - 1) * 2

        # assert torch.all(torch.eq(priors[..., 2], torch.ones(1, 1, nA, nH, nW)))
        # assert torch.all(torch.eq(priors[..., 3], torch.ones(1, 1, nA, nH, nW).mul(2)))

    def test_point_form_conversion(self):
        FloatTensor = torch.FloatTensor
        nB = 5
        nT = 5
        nA = 5
        nH = 44
        nW = 50
        priors = torch.zeros(1, 1, nA, nH, nW, 4)
        grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        priors[..., 0] = grid_x.expand(-1, -1, nA, -1, -1).contiguous()
        priors[..., 1] = grid_y.expand(-1, -1, nA, -1, -1).contiguous()
        anchor_w = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1)
        anchor_l = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1).mul(2)

        priors[..., 2] = anchor_l
        priors[..., 3] = anchor_w

        new_priors = point_form_fafe(priors)
        print(priors.shape)
        print(new_priors.shape)

        print(priors[0][0][:][0][0])
        print(new_priors[0][0][:][0][0])

    def test_point_form_ground_truth(self):
        ground_truth = torch.rand((2, 5, 50, 6))
        # x, y, l, w, rot, class

        point_from_gt = point_form_fafe(ground_truth[..., 0:4])
        print(point_from_gt.shape)

    def test_intersection(self):
        FloatTensor = torch.FloatTensor
        nB = 5
        nT = 5
        nA = 5
        nH = 44
        nW = 50
        nGT = 3
        priors = torch.zeros(1, 1, nA, nH, nW, 4)
        grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view([1, 1, 1, nH, nW]).type(
            FloatTensor)
        priors[..., 0] = grid_x.expand(-1, -1, nA, -1, -1).contiguous()
        priors[..., 1] = grid_y.expand(-1, -1, nA, -1, -1).contiguous()
        anchor_w = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1)
        anchor_l = torch.arange(1, float(nA + 1)).view(1, 1, nA, 1, 1).mul(2)

        priors[..., 2] = anchor_l
        priors[..., 3] = anchor_w

        point_form_priors = point_form_fafe(priors)

        ground_truth = torch.rand(nGT, 4).mul(40)

        point_form_gt = point_form_fafe(ground_truth)
        gt = point_form_gt.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(nA, nH, nW, -1, -1)
        prior = point_form_priors.squeeze(0).squeeze(0).unsqueeze(3).expand(-1, -1, -1, nGT, -1)
        # lollie = our_intersect(gt, prior)
        # overlap = our_jaccard(gt, prior)
        match = our_match(truths=gt, priors=prior)

    def test_center(self):
        reg_targets = torch.rand((3, 6))
        z_center = -1
        center_size_3d(reg_targets, z_center)
