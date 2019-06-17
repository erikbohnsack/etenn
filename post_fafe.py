import torch
from scipy.spatial.distance import cdist
import numpy as np
from fafe_utils.fafe_utils import too_close


class PostFafe:
    """
    data_state = list of lists.     [[t_0 data],
                                    [t_1 data],
                                    ...,
                                    [t_4 data]]
                where t_0 data is a list of size num_conseq_frames, containing inference results for each predicted
                time frame.
    """

    def __init__(self, input_config, post_config, device):

        self.nT = input_config.num_conseq_frames
        self.iou_threshold = post_config.iou_threshold
        self.reason_distance_threshold = post_config.reason_distance_threshold
        self.det2pred_distance_threshold = post_config.det2pred_distance_threshold
        self.over_time_distance_threshold = post_config.over_time_distance_threshold
        self.confidence_threshold = post_config.confidence_threshold
        self.device = device
        self.data_state = []
        self.object_state = {}
        self.id_count = 0
        self.verbose = post_config.verbose

        print("Post Fafe, num_conseq_frames: {}".format(self.nT))

    def is_new(self):
        return len(self.data_state) == 0 and not bool(self.object_state)

    def new_data(self, data):
        """

        :param data:
        :return:
        """

        checked_data = []
        # Remove detections not in the FOV
        for timed_data in data:
            if len(timed_data) == 0:
                checked_data.append(timed_data)
                continue
            timed_data = too_close(timed_data, 1)
            checked_data.append(torch.cat(timed_data, dim=0))
        self.data_state.insert(0, checked_data)
        if len(self.data_state) > self.nT:
            self.data_state.pop()


    def match_n_infer(self, timestep):
        """
        Matches bounding boxes over earlier timesteps predictions with current detections.
        This is done for given _timestep_ in time. So timestep = 0 outputs the joinly reasoned
        inference detections for the current timestep. timestep = 1 equals jointly reasoned
        predictions one step in the future.
        :param timestep: int. Must be in interval [0, self.nT]
        :return: jointly_reasoned.  List of tensors, holding inference detection results.
        """
        data = []

        # Loop over stored data states. Min function to check if we have
        # just started and not stored. (self.nT - timestep) is because if
        # we want to compute for timestep 2, we only want to grab data from
        # predictions within the reasonable timeframe.
        for i in range(min(self.nT - timestep, len(self.data_state))):
            j = i + timestep
            try:
                ij_data = self.data_state[i][j]
                data.append(ij_data)
            except IndexError as e:
                print("Index error! No data found here, i:{}, j:{}, timestep:{}".format(i, j, timestep))
                raise e


        # TODO: Double check that this doesnt kick in
        if not data:
            raise ValueError("DATA EMPTY")

        # Remove empty ones from data.
        checked_data = [item for item in data if len(item)]
        if not checked_data:
            return checked_data

        stacked_data = torch.cat(checked_data, dim=0).to('cpu')

        dist = cdist(stacked_data[:, 0:2], stacked_data[:, 0:2], metric='euclidean')
        dist_bool = (dist < self.reason_distance_threshold)
        nonzero_bool = dist != 0.
        # TODO: IoU instead of cdist
        iou_bool = np.multiply(dist_bool, nonzero_bool)

        # Check matching boxes
        avoid = []  # If data point already taken by matched point, ignore
        jointly_reasoned = []  # Store jointly_reasoned data.

        # Loop over rows in matching matrix, i.e.
        for i in range(iou_bool.shape[0]):

            if i in avoid:
                continue

            # Check if there are any matches
            if iou_bool[i].any():
                match = [i] + np.argwhere(iou_bool[i]).reshape(-1, ).tolist()
                avoid.extend(match)  # Avoid these in the future.
                mean_ = torch.mean(stacked_data[match], dim=0).squeeze(0)

                # If mean class score higher than threshold, add to output
                #if mean_[5] > self.confidence_threshold:
                jointly_reasoned.append(mean_)

            # If not any matches, add as it is if class score high enough.
            else:
                avoid.append(i)
                #if stacked_data[i][5] > self.confidence_threshold:
                jointly_reasoned.append(stacked_data[i].squeeze(0))

        return jointly_reasoned

    def id_reasoned(self, reasoned_state):
        """
        Match the reasoned state so that the current timestep has connected the predictions.

        :param reasoned_state List of lists. Outmost list is over time. nT long.
                Inner list is the reasoned measurements for one timestep. Contains tensors or empty list.


        :return: objects
            List of lists. Each object is a list with length self.nT and
            at least one tensor. Each list item for one object is one predicted state.
            If there is no predicted state, item is None.
        """
        assert len(reasoned_state) == self.nT, 'Len of reasoned state not nT'

        # Create objects for states in the first timestep.
        objects = []

        # Make sure there are measurements for time=0, otherwise don't create any new objects. Return empty list
        if reasoned_state[0]:
            for i in range(len(reasoned_state[0])):
                objects.append([reasoned_state[0][i]])
        else:
            return objects

        # Match new objects for time = 0 with predictions.
        # cdist current with predicted
        for i in range(self.nT - 1):

            # If empty tensor for prediction i + 1, continue.
            if len(reasoned_state[i + 1]) == 0:

                # Append None to all objects for this timestep.
                for k in range(len(objects)):
                    objects[k].append(None)
                continue

            dist = cdist(torch.stack(reasoned_state[0])[:, 0:2], torch.stack(reasoned_state[i + 1])[:, 0:2],
                         metric='euclidean')
            dist_bool = (dist < self.det2pred_distance_threshold)

            # loop over rows in matrix, i.e. loop over current state for different objects, checking i-th prediction.
            for j in range(dist_bool.shape[0]):

                # If match with pred, add to object prediction state
                if dist_bool[j].any():
                    indices = np.argwhere(dist_bool[j]).reshape(-1, ).tolist()
                    # If it matches with more than 1, just take mean of the matched.
                    if len(indices) > 1:
                        mean_ = torch.mean(torch.stack(reasoned_state[i + 1])[indices], dim=0).squeeze(0)
                        objects[j].append(mean_)
                    else:
                        objects[j].append(torch.stack(reasoned_state[i + 1])[indices].squeeze(0))

                # if no match, set None
                else:
                    objects[j].append(None)

        return objects

    def match_objects(self, objects):
        """
        Match over time, i.e. old objects to new objects to see if they are the same.
        Match current dict of objects in self.object_state with new object.

        :param objects: List of objects
        :return:
        """

        object_states = torch.stack([obj[0] for obj in objects])

        # print("Object states: type: {} shape: {}".format(type(object_states), object_states.shape))
        # Loop over old objects.
        new_object_state = {}
        used_new_states = []
        for idx, obj in self.object_state.items():
            # Check distance t
            if type(obj) is list:
                obj_to_match = obj[0]
            else:
                obj_to_match = obj

            try:
                dist = cdist(obj_to_match.unsqueeze(0)[:, 0:2], object_states[:, 0:2])
            except IndexError as e:
                print("idx: {} obj: {}".format(idx, len(obj)))
                print("idx: {} obj: {}".format(idx, obj))
                print("{}".format(obj[0].unsqueeze(0)[:, 0:2]))
                raise e
            dist_bool = (dist < self.over_time_distance_threshold)

            if dist_bool.any():
                indices = np.argwhere(dist_bool[0]).reshape(-1, )

                # Check if the new objects match has already been used.
                if any(i in used_new_states for i in indices):
                    continue

                if len(indices) > 1:
                    # need to mean all timesteps.
                    first__ = [objects[j][0] for j in indices if objects[j][0] is not None]
                    sec__ = [objects[j][1] for j in indices if objects[j][1] is not None]
                    third__ = [objects[j][2] for j in indices if objects[j][2] is not None]
                    four__ = [objects[j][3] for j in indices if objects[j][3] is not None]
                    fif__ = [objects[j][4] for j in indices if objects[j][4] is not None]

                    # Need to check if there are elements in list since it can be empty.
                    if first__:
                        first_ = torch.stack(first__)
                        first = torch.mean(first_, dim=0).squeeze(0)
                    else:
                        first = None

                    if sec__:
                        sec_ = torch.stack(sec__)
                        sec = torch.mean(sec_, dim=0).squeeze(0)
                    else:
                        sec = None

                    if third__:
                        third_ = torch.stack(third__)
                        third = torch.mean(third_, dim=0).squeeze(0)
                    else:
                        third = None

                    if four__:
                        four_ = torch.stack(four__)
                        four = torch.mean(four_, dim=0).squeeze(0)
                    else:
                        four = None

                    if fif__:
                        fif_ = torch.stack(fif__)
                        fif = torch.mean(fif_, dim=0).squeeze(0)
                    else:
                        fif = None

                    state_ = [first, sec, third, four, fif]
                else:
                    state_ = objects[indices[0]]
                used_new_states.extend(indices)
                new_object_state[idx] = state_

        # Check if new state has not been matched. In that case, add new.
        for i in range(len(objects)):
            if i not in used_new_states:
                new_object_state[self.id_count] = objects[i]
                self.id_count += 1

        self.object_state = new_object_state

    def set_object_state(self, objects):
        """

        :param objects: List of objects
        :return:
        """
        for obj in objects:
            self.object_state[self.id_count] = obj
            self.id_count += 1

    def step_with_no_new_objects(self):
        """
        Take timestep if there are no new objects.
        :return:
        """
        new_object_state = {}

        # Loop over present objects
        for idx, obj in self.object_state.items():
            # If there is predictions for current object,
            #
            # append None to end of list,
            obj.append(None)

            # move all elements to the left.
            obj.pop(0)

            # Keep same ID and
            if not all(x is None for x in obj):
                new_object_state[idx] = obj

        self.object_state = new_object_state
    def __call__(self, fafe_data):
        """
        Insider trading.

        :param fafe_data: Lists of tensors
        :return:
        """

        # TODO: Double check that this works in general
        if fafe_data is None:
            return

        self.new_data(fafe_data)
        reasoned_state = []
        for timestep in range(self.nT):
            output = self.match_n_infer(timestep=timestep)
            reasoned_state.append(output)

        if self.verbose:
            print("Len of reasoned state: {}".format(len(reasoned_state)))
        new_objects = self.id_reasoned(reasoned_state)

        # TODO: Make sure to grab empty list

        if self.verbose:
            print("Len of new objects: {}".format(len(new_objects)))

        # Match new objects with old object state
        if self.object_state:
            if new_objects:

                self.match_objects(new_objects)
            else:
                self.step_with_no_new_objects()

        # Else if it's empty, just add new objects.
        else:
            if self.verbose:
                print("Storing new objects")
            self.set_object_state(new_objects)

        return
