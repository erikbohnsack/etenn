import torch
import torch.nn as nn
from .box_utils import point_form_fafe, our_match
import torch.nn.functional as F


class FafeLoss(nn.Module):
    def __init__(self, input_config, train_config, loss_config, device=None):
        super().__init__()
        self._nA = input_config.num_anchors
        self._num_det_targets = input_config.num_classes + 1
        self._num_classes = input_config.num_classes
        self._num_conseq_frames = input_config.num_conseq_frames
        self._num_reg_targets = input_config.num_reg_targets
        self._fov_width = input_config.y_max - input_config.y_min
        self._fov_height = input_config.x_max - input_config.x_min

        self.conf_threshold = loss_config.confidence_threshold
        self.lower_match_threshold = loss_config.lower_match_threshold
        self.higher_match_threshold = loss_config.higher_match_threshold

        if loss_config.regression_loss == 'MSE':
            self.reg_loss_fn = nn.MSELoss(reduction='mean')
        elif loss_config.regression_loss == 'SmoothL1':
            self.reg_loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            print('\n** USING [SmoothL1Loss] REGRESSION LOSS PER DEFAULT **\n')
            self.reg_loss_fn = nn.SmoothL1Loss(reduction='mean')

        if loss_config.euler_loss == 'MSE':
            self.euler_loss_fn = nn.MSELoss(reduction='mean')
        elif loss_config.euler_loss == 'SmoothL1':
            self.euler_loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            print('\n** USING [SmoothL1Loss] EULER LOSS PER DEFAULT **\n')
            self.euler_loss_fn = nn.SmoothL1Loss(reduction='mean')

        self._lambda = loss_config.lambda_time_decay

        self.softmax = nn.Softmax(dim=-1)
        self.alpha_factor = loss_config.alpha_factor
        self.gamma = loss_config.gamma
        self.regression_beta = loss_config.regression_beta
        self.euler_beta = loss_config.euler_beta
        self.class_beta = loss_config.class_beta

        if device is None:
            self.device = torch.device("cuda:" + str(train_config.cuda_device))
        else:
            self.device = device

        self.FloatTensor = torch.cuda.FloatTensor if train_config.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if train_config.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if train_config.use_cuda else torch.ByteTensor

        scaled_anchors = self.FloatTensor([(a_l, a_w) for a_l, a_w in input_config.anchors]).to(self.device)
        self.anchor_l = scaled_anchors[:, 0:1].view((1, 1, self._nA, 1, 1))
        self.anchor_w = scaled_anchors[:, 1:2].view((1, 1, self._nA, 1, 1))

        self.grid_x, self.grid_y = None, None
        self.point_form_priors = None

    def __call__(self,
                 out_detection,
                 out_regression,
                 ground_truth,
                 verbose,
                 **params):
        """Call the loss function.

        Args:
          prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
          target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
          ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
          scope: Op scope name. Defaults to 'Loss' if None.
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: a tensor representing the value of the loss function.
        """

        return self.forward(out_detection, out_regression, ground_truth, verbose)

    def forward(self, out_detection, out_regression, ground_truth, verbose):
        """

        out_detection: (#batches x #output_filters_detection x nH x nW)
        out_regression: (#batches x #output_filters_regression x nH x nW)
        ground_truth: (#batches x #conseq_frames x # max targets forever x # ground truth dimensions)
        """
        nA = self._nA  # num_anchors = 5
        nB = out_detection.data.size(0)  # batch_size
        nT = self._num_conseq_frames
        nH = out_detection.data.size(2)
        nW = out_detection.data.size(3)

        # Splitting up the channel dimension to time, anchors and targets
        # detection reformat to:
        # [#batches, #conseq_frames, #anchors, height, width, detection_targets]
        detection = out_detection.view(nB, nA, self._num_det_targets, self._num_conseq_frames, nH, nW)\
                                 .permute(0, 3, 1, 4, 5, 2).contiguous()

        detection = self.softmax(detection)

        # regression reformat to:
        # [#batches, #conseq_frames, #anchors, height, width, regression_parameter_outputs]
        regression = out_regression.view(nB, nA, self._num_reg_targets, self._num_conseq_frames, nH, nW)\
                                  .permute(0, 3, 1, 4, 5, 2).contiguous()

        if verbose:
            print('Detection: {}'.format(detection.requires_grad))
            print('Regression: {}'.format(regression.requires_grad))
            print('{}'.format('~' * 10))
            print('Detection shape: {}'.format(detection.shape))
            print('Regression shape: {}'.format(regression.shape))

        # Get outputs; the scaling factor of each anchor box
        # Center x scale. Multiplication with the voxel height
        t_x = torch.sigmoid(regression[..., 0]).mul(self._fov_height / nH)
        # Center y scale. Multiplication with the voxel width
        t_y = torch.sigmoid(regression[..., 1]).mul(self._fov_width / nW)
        # Width scale
        t_w = torch.sigmoid(regression[..., 2])
        # Length scale
        t_l = torch.sigmoid(regression[..., 3])

        # Calculate offsets for each grid
        if self.grid_x is None or self.grid_y is None:
            self.grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view(
                [1, 1, 1, nH, nW]).type(self.FloatTensor).to(self.device)
            self.grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view(
                [1, 1, 1, nH, nW]).type(self.FloatTensor).to(self.device)

        # Add offset and scale with anchors
        # reg_boxes :
        #        shape : [#batches, #conseq_frames, #anchors, height, width, regression_targets]
        #        last dim contains : [x_pos, y_pos, bbox_length, bbox_width, bbox_rotation]
        reg_boxes = self.FloatTensor(regression[..., :4].shape).to(self.device)
        reg_boxes[..., 0] = t_x + self.grid_x
        reg_boxes[..., 1] = t_y + self.grid_y
        reg_boxes[..., 2] = torch.exp(t_l).mul(self.anchor_l)
        reg_boxes[..., 3] = torch.exp(t_w).mul(self.anchor_w)

        # These are the values we want to train for so set reg_boxes to True...
        # But only when training :)
        if detection.requires_grad:
            reg_boxes.requires_grad_(True)

        if verbose:
            print('{}'.format('~' * 10))
            print('reg_boxes: {}'.format(reg_boxes.shape))
            print("\tx reg box: {}".format(reg_boxes[..., 0].shape))
            print("\ty reg box: {}".format(reg_boxes[..., 1].shape))
            print("Reg box example: {}".format(reg_boxes[0][0][0][0][0][:]))

        # Create priors for each element that can be matched with ground truths
        if self.point_form_priors is None:
            priors = self.FloatTensor(nA, nH, nW, 4).to(self.device)
            priors[..., 0] = self.grid_x.expand(-1, -1, nA, -1, -1).contiguous()
            priors[..., 1] = self.grid_y.expand(-1, -1, nA, -1, -1).contiguous()
            priors[..., 2] = self.anchor_l
            priors[..., 3] = self.anchor_w
            # Reformat last dims from x,y,w,l to x1,y1,x2,y2
            self.point_form_priors = point_form_fafe(priors)

        if verbose:
            print('{}\nPriors: {}'.format('~' * 10, self.point_form_priors.shape))
            print('Prior example: {}'.format(self.point_form_priors[0][0][0][:]))

        reg_loss, euler_loss, classification_loss = 0, 0, 0
        n_gt, n_objects, n_background, n_correct = 0, 0, 0, 0

        for batch_index in range(nB):
            for time_index in range(nT):
                # Fetch and re-format ground truths
                non_zero_gt_mask = ground_truth[batch_index][time_index].sum(dim=1) != 0
                non_zero_gt = ground_truth[batch_index][time_index][non_zero_gt_mask]
                point_form_gt = point_form_fafe(non_zero_gt)
                _ngt = non_zero_gt.shape[0]
                n_gt += _ngt
                if verbose:
                    print('{}\nNON-ZERO GT\n\tNon-zero GT shape: {}'.format('~' * 10, non_zero_gt.shape))
                    print('\tNon-zero GT example: {}'.format(non_zero_gt))

                # Reshape both gt and priors to [nA, nH, nW, n_gt, dim_points (4)] to easen up calculations
                gt = point_form_gt.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(nA, nH, nW, -1, -1)
                prior = self.point_form_priors.unsqueeze(3).expand_as(gt)
                if verbose:
                    print('{}\nPOINT FORM'.format('~' * 10))
                    print('\tGT shape: {}'.format(gt.shape))
                    print('\tGT example: {}'.format(gt[0][0][0][:]))
                    print('\tPrior shape: {}'.format(prior.shape))
                    print('\tPrior example: {}'.format(prior[0][0][0][:]))

                # Match anchors/priors with ground truths if there are any ground truths...
                if _ngt > 0:
                    mask, gt_mask, negative_mask = our_match(truths_pf=gt, priors=prior,
                                                             lower_threshold=self.lower_match_threshold,
                                                             higher_threshold=self.higher_match_threshold)
                    if verbose:
                        print('~' * 10)
                        print('MASK')
                        print('\tMask shape: {}'.format(mask.shape))
                        print("\tMask sum: {}".format(mask.sum()))
                        print('\tGT mask: {}'.format(gt_mask))
                    regression_truths = non_zero_gt[gt_mask, 0:4]
                    regression_predictions = reg_boxes[batch_index][time_index][mask]

                    #reg_pred_x =  reg_x[batch_index][time_index][mask]

                    # [#batches, #conseq_frames, #anchors, height, width, regression_parameter_outputs]
                    gt_re = torch.cos(non_zero_gt[gt_mask, 4])
                    gt_im = torch.sin(non_zero_gt[gt_mask, 4])
                    reg_im = regression[batch_index][time_index][mask][:, 5]
                    reg_re = regression[batch_index][time_index][mask][:, 4]

                    # Lambda ^ time_index in order to make predictions less and less important the
                    # further away from current time we are

                    #reg_loss_x += self._lambda ** time_index * \
                    #            self.reg_loss_fn(reg_pred_x, regression_truths[..., 0]) / _ngt

                    reg_loss += self._lambda ** time_index * \
                                      self.reg_loss_fn(regression_predictions, regression_truths) / _ngt
                    euler_loss += self._lambda ** time_index * \
                                  (self.euler_loss_fn(reg_im, gt_im) + self.euler_loss_fn(reg_re, gt_re)) / _ngt

                    if verbose:
                        print('~' * 10)
                        print('Regression:')
                        print('\tnReg: {}\n\tnGT: {}'.format(len(regression_predictions), len(regression_truths)))
                        print("\tregression_predictions\n\t\t{}".format(regression_predictions))
                        print("\tregression_truths\n\t\t{}".format(regression_truths))
                        print('~' * 10)
                        print('Angle Regression')
                        print('\tGT Re: {}'.format(gt_re))
                        print('\tReg Re: {}'.format(reg_re))
                        print('\tGT Im: {}'.format(gt_im))
                        print('\tReg Im: {}'.format(reg_im))
                        print('~' * 10)
                        print('Loss')
                        print('\tCurrent reg: {}'.format(self._lambda ** time_index * \
                                      self.reg_loss_fn(regression_predictions, regression_truths) / _ngt))
                        print('\tCurrent eul: {}'.format(self._lambda ** time_index * \
                                  (self.euler_loss_fn(reg_im, gt_im) + self.euler_loss_fn(reg_re, gt_re)) / _ngt))
                        print('\tReg_loss: {}'.format(reg_loss))
                        print('\tEul_loss: {}'.format(euler_loss))

                    if verbose:
                        print('{}\nRegression Truths: {}'.format('~' * 10, regression_truths.shape))
                        print('\tReg truths example: {}'.format(regression_truths[-1]))
                        print('{}\nRegression predictions: {}'.format('~' * 10, regression_predictions.shape))
                        print('\tReg Pred example: {}'.format(regression_predictions[-1]))
                        print('~' * 10)
                        print('sl1l: {}'.format(self.reg_loss_fn(regression_predictions, regression_truths)))


                # If not having any ground truths we take the ones that are very faulty, i.e. having object class score
                # larger than a threshold
                else:
                    mask = detection[batch_index, time_index, ..., 1] > self.conf_threshold
                    negative_mask = torch.ones_like(mask) - mask

                if verbose:
                    print('{}'.format('~' * 10))
                    print('[iB={}, iT={}]'.format(batch_index, time_index))
                    print('\tNon-zero-gt: {}'.format(non_zero_gt.shape))
                    print("\tGT: \t{}".format(gt.shape))
                    print("\tPrior: \t{}".format(prior.shape))
                    print('\tMask: \t{}'.format(mask.shape))
                    print('\tMask: \t{}'.format(type(mask)))
                    print('\tSmooth_L1: {}'.format(reg_loss))
                    print('\tSmooth_L1: {}'.format(type(reg_loss)))
                    print('\tDetection: {}'.format(detection.shape))
                    print('\tDetection[batch_index]: {}'.format(detection[batch_index].shape))
                    print('\tDetection[batch_index, time_index]: {}'.format(detection[batch_index, time_index].shape))
                    print('\tDetection[batch_index, time_index, mask]: {}'.format(
                        detection[batch_index, time_index, mask].shape))
                    print('\tDetection[batch_index, time_index][mask]: {}'.format(
                        detection[batch_index, time_index][mask].shape))

                # If we don't want to pick any elements with mask. Manually set the probability and class to zero in
                # order to not mess up .max(dim=1)
                if mask.sum() != 0:
                    detected_probability, detected_class = detection[batch_index, time_index][mask].max(dim=1)
                else:
                    detected_probability = self.FloatTensor([0]).to(self.device)
                    detected_class = self.FloatTensor([0]).to(self.device)
                if negative_mask.sum() != 0:
                    undetected_probability, undetected_class = detection[batch_index, time_index, negative_mask].max(
                        dim=1)
                else:
                    undetected_probability = self.FloatTensor([0]).to(self.device)
                    undetected_class = self.FloatTensor([0]).to(self.device)

                if verbose:
                    print('\tNegative_mask : {}'.format(negative_mask.shape))
                    print('\tDetected classes: {}'.format(detected_class))
                    print('\tDetected probabilities: {}'.format(detected_probability))
                    print('\tUndetected classes: {}'.format(undetected_class.shape))
                    print('\tUndetected probabilities: {}'.format(undetected_probability.shape))
                    print("\tMask sum: {}".format(mask.sum()))
                    print("\tNeg mask sum: {}".format(negative_mask.sum()))

                # TODO: fix for multiple classes
                if out_detection.is_cuda:
                    prob_tensor = torch.zeros(nA, nH, nW, device=self.device)
                else:
                    prob_tensor = torch.zeros(nA, nH, nW)

                if _ngt > 0:
                    prob_tensor[mask] = detection[batch_index, time_index, mask][:, 1]
                else:
                    prob_tensor[mask] = detection[batch_index, time_index, mask][:, 0]

                prob_tensor[negative_mask] = detection[batch_index, time_index, negative_mask][:, 0]

                alpha = mask.float().mul(self.alpha_factor) + negative_mask.float().mul(1 - self.alpha_factor)

                n_objects += (detection[batch_index, time_index, :, :, :, :][..., 1] >= 0.5).sum()
                # n_background += (detection[batch_index, time_index, :, :, :, :][... , 0] > 0.5).sum()
                n_correct += (detected_class != 0).sum()

                if verbose:
                    print('mask: {}'.format(mask.requires_grad))
                    print('prob_tensor: {}'.format(prob_tensor.requires_grad))

                BCE_loss = F.binary_cross_entropy(prob_tensor, torch.ones_like(prob_tensor), weight=None,
                                                  reduction='none')
                F_loss = alpha * (1 - torch.exp(-BCE_loss)) ** self.gamma * BCE_loss
                classification_loss += self._lambda ** time_index * F_loss.mean()

                if verbose:
                    print("\tAlpha: {}".format(alpha.shape))
                    print('\tProbability tensor: {}'.format(prob_tensor.shape))
                    print('\tProbability tensor: {}'.format(prob_tensor.shape))
                    print("\tBCE_loss {}".format(BCE_loss.shape))
                    print("\tF_loss {}".format(F_loss.shape))
                    print("\tClassification Loss: {}".format(classification_loss))

        scaled_reg = self.regression_beta * reg_loss
        scaled_euler = self.euler_beta * euler_loss
        scaled_class = self.class_beta * classification_loss

        total_loss = scaled_class + scaled_reg + scaled_euler

        recall = float(n_correct) / float(n_gt) if n_gt else 1
        # min(. , 10) due to prevent precision from blowing up and ruining plots...
        precision = min(float(n_correct) / float(n_objects), 10) if n_objects else 0

        if verbose:
            print('{}'.format('~' * 10))
            print('nB:  {}'.format(nB))
            print('nT:  {}'.format(nT))
            print('NumGT:  {}'.format(n_gt))
            print('NumObj: {}'.format(n_objects))
            # print('NumBg:  {}'.format(n_background))
            # print('Total:  {}'.format(n_objects + n_background))
            print('NumCorrect: {}'.format(n_correct))
            print('Recall: {}'.format(recall))
            print('Precision: {}'.format(precision))
            print('{}'.format('~' * 10))
            print('Total Loss: {}'.format(total_loss))
            print('beta * Reg Loss: {}'.format(self.regression_beta * reg_loss))
            print('Cls Loss: {}'.format(classification_loss))
            print('Scaled Reg Loss: {}'.format(scaled_reg))
            print(scaled_reg)
            print('Scaled euler Loss: {}'.format(scaled_euler))
            print(scaled_euler)
            print('Scaled class Loss: {}'.format(scaled_class))
            print(scaled_class)

        # Make sure to only save the value (and not the entire backprop grad_fn)
        scaled_reg = scaled_reg.item() if scaled_reg != 0 else 0
        scaled_euler = scaled_euler.item() if scaled_euler != 0 else 0
        scaled_class = scaled_class.item() if scaled_class != 0 else 0

        return total_loss, recall, precision, scaled_reg, scaled_euler, scaled_class
