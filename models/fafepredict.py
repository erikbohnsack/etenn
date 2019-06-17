import torch.nn as nn
import torch


class FafePredict(nn.Module):
    def __init__(self, input_config, eval_config):
        super().__init__()
        self.conf_threshold = eval_config.confidence_threshold
        self.verbose = eval_config.verbose
        self._nA = eval_config.num_anchors
        self._num_conseq_frames = input_config.num_conseq_frames
        self._num_det_targets = input_config.num_classes + 1
        self._num_reg_targets = input_config.num_reg_targets
        self.softmax = nn.Softmax(dim=-1)
        self._fov_width = input_config.y_max - input_config.y_min
        self._fov_height = input_config.x_max - input_config.x_min
        self.anchors = eval_config.anchors
        self.grid_x = None
        self.grid_y = None
        self.FloatTensor = torch.cuda.FloatTensor if eval_config.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if eval_config.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if eval_config.use_cuda else torch.ByteTensor

    def __call__(self, out_detection, out_regression):
        return self.forward(out_detection=out_detection, out_regression=out_regression)

    def forward(self, out_detection, out_regression):
        """

        out_detection: (#batches x #output_filters_detection x nH x nW)
        out_regression: (#batches x #output_filters_regression x nH x nW)
        """
        nA = self._nA  # num_anchors = 5
        nB = out_detection.data.size(0)  # batch_size
        nT = self._num_conseq_frames
        nH = out_detection.data.size(2)
        nW = out_detection.data.size(3)



        # Splitting up the channel dimension to time, anchors and targets
        # detection reformat to:
        # [#batches, #conseq_frames, #anchors, height, width, detection_targets]
        detection = out_detection.view(nB, nA, self._num_det_targets, self._num_conseq_frames, nH, nW).permute(0, 3, 1,
                                                                                                               4, 5,
                                                                                                               2).contiguous()
        detection = self.softmax(detection)

        # regression reformat to:
        # [#batches, #conseq_frames, #anchors, height, width, regression_parameter_outputs]
        regression = out_regression.view(nB, nA, self._num_reg_targets, self._num_conseq_frames, nH, nW).permute(0, 3,
                                                                                                                 1, 4,
                                                                                                                 5,
                                                                                                                 2).contiguous()

        if self.verbose:
            print('{}'.format('~' * 10))
            print('Detection shape: {}'.format(detection.shape))
            print('Regression shape: {}'.format(regression.shape))

        # Get outputs; the scaling factor of each anchor box
        t_x = torch.sigmoid(regression[..., 0]).mul(
            self._fov_height / nH)  # Center x scale. Multiplication with the voxel height
        t_y = torch.sigmoid(regression[..., 1]).mul(
            self._fov_width / nW)  # Center y scale. Multiplication with the voxel width
        #t_w = torch.sigmoid(regression[..., 2])  # Width scale
        #t_l = torch.sigmoid(regression[..., 3])  # Length scale
        t_w = regression[..., 2]  # Width scale
        t_l = regression[..., 3]  # Length scale

        # Imaginary divided by real.
        angle = torch.atan2(regression[..., 5], regression[..., 4])

        # Calculate offsets for each grid
        if self.grid_x is None or self.grid_y is None:
            self.grid_x = torch.arange(0, float(nH)).mul(self._fov_height / nH).repeat(nW, 1).view(
                [1, 1, 1, nH, nW]).type(self.FloatTensor)
            self.grid_y = torch.arange(-nW / 2, nW / 2).mul(self._fov_width / nW).repeat(nH, 1).view(
                [1, 1, 1, nH, nW]).type(self.FloatTensor)

        scaled_anchors = self.FloatTensor([(a_l, a_w) for a_l, a_w in self.anchors])
        anchor_l = scaled_anchors[:, 0:1].view((1, 1, nA, 1, 1))
        anchor_w = scaled_anchors[:, 1:2].view((1, 1, nA, 1, 1))
        # anchor_r = scaled_anchors[:, 2:3].view((1, nA, 1, 1, 1))

        # Add offset and scale with anchors
        # reg_boxes :
        #        shape : [#batches, #conseq_frames, #anchors, height, width, regression_targets]
        #        last dim contains : [x_pos, y_pos, bbox_length, bbox_width, bbox_rotation]
        reg_boxes = self.FloatTensor(regression[..., :5].shape)
        reg_boxes[..., 0] = t_x.data + self.grid_x
        reg_boxes[..., 1] = t_y.data + self.grid_y
        reg_boxes[..., 2] = torch.exp(t_l.data) * anchor_l
        reg_boxes[..., 3] = torch.exp(t_w.data) * anchor_w
        reg_boxes[..., 4] = angle.data  # + anchor_r
        inference_results = []
        for batch_index in range(nB):
            time_inference = []
            for time_index in range(nT):
                # Get the probabilities of having an object in each feature map element
                # for each anchor box
                object_prob = detection[batch_index, time_index, ..., 1]

                # Create a mask where the probabilities are above a certain threshold
                mask_object_prob = object_prob > self.conf_threshold

                # Find which anchor boxes have the highest probabilities of object for each
                # elemnt in the feature map
                best_anchor_prob, best_anchor_index = object_prob.max(dim=0, keepdim=False)

                # Go back and set the mask to 0 where the anchor box is not the one with
                # highest probability of containing an object
                for anchor_index in range(len(scaled_anchors)):
                    anchor_mask = best_anchor_index != anchor_index
                    mask_object_prob[anchor_index][anchor_mask] = 0

                #best_pred, best_pred_index = detection[batch_index, time_index].max(dim=-1)
                ## Mask if confident and it's not background.
                #mask1 = best_pred > self.conf_threshold
                #mask2 = best_pred_index != 0
                #mask = mask1.mul(mask2)
                #preds = best_pred_index[mask]

                outputs = reg_boxes[batch_index, time_index, mask_object_prob]
                preds = object_prob[mask_object_prob]

                #print('outputs: {}'.format(outputs))
                #print('preds: {}'.format(preds))

                time_inference.append(torch.cat((outputs, preds.unsqueeze(1).float()), dim=1))
            inference_results.append(time_inference)
        return inference_results
