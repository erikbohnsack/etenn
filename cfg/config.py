import numpy as np


class Config:
    def __init__(self, conf):
        self._config = conf

    def get_property(self, property_name):
        if property_name not in self._config.keys():
            return None
        return self._config[property_name]

    def __repr__(self):
        return "Config: {}".format(self._config)


class InputConfig(Config):

    @property
    def bev_shape(self):
        x_shape = np.int(
            np.round((self.get_property("x_max") - self.get_property("x_min")) / self.get_property("x_grid_size")))
        y_shape = np.int(
            np.round((self.get_property("y_max") - self.get_property("y_min")) / self.get_property("y_grid_size")))
        z_shape = np.int(
            np.round((self.get_property("z_max") - self.get_property("z_min")) / self.get_property("z_grid_size")))
        if self.get_property("num_point_features") == 3:
            return x_shape, y_shape, z_shape
        else:
            return x_shape, y_shape, z_shape, 1

    @property
    def x_min(self):
        return self.get_property("x_min")

    @property
    def x_max(self):
        return self.get_property("x_max")

    @property
    def x_grid_size(self):
        return self.get_property("x_grid_size")

    @property
    def y_grid_size(self):
        return self.get_property("y_grid_size")

    @property
    def z_grid_size(self):
        return self.get_property("z_grid_size")

    @property
    def z_pillar_size(self):
        return self.get_property("z_pillar_size")

    @property
    def y_min(self):
        return self.get_property("y_min")

    @property
    def y_max(self):
        return self.get_property("y_max")

    @property
    def z_min(self):
        return self.get_property("z_min")

    @property
    def z_max(self):
        return self.get_property("z_max")

    @property
    def num_point_features(self):
        return self.get_property("num_point_features")

    @property
    def num_conseq_frames(self):
        return self.get_property("num_conseq_frames")

    @property
    def num_reg_targets(self):
        return self.get_property("num_reg_targets")

    @property
    def num_classes(self):
        return self.get_property("num_classes")

    @property
    def num_anchors(self):
        return len(self.get_property("anchors"))

    @property
    def dim_gt_targets(self):
        return self.get_property("dim_gt_targets")

    @property
    def max_targets_forever(self):
        return self.get_property("max_targets_forever")

    @property
    def anchors(self):
        return self.get_property("anchors")

    @property
    def get_labels(self):
        return self.get_property("get_labels")

    @property
    def z_center(self):
        return self.get_property("z_center")

    @property
    def z_height(self):
        return self.get_property("z_height")

    @property
    def dt(self):
        return self.get_property("dt")

    @property
    def pp_max_points(self):
        return self.get_property("pp_max_points")

    @property
    def pp_max_voxels(self):
        return self.get_property("pp_max_voxels")

    @property
    def pp_reverse_index(self):
        return self.get_property("pp_reverse_index")

    @property
    def pp_num_filters(self):
        return tuple(self.get_property("pp_num_filters"))

    @property
    def use_pp(self):
        return self.get_property("use_pp")

    @property
    def pp_use_norm(self):
        return self.get_property("pp_use_norm")

    @property
    def pp_with_distance(self):
        return self.get_property("pp_with_distance")

    @property
    def pp_verbose(self):
        return self.get_property("pp_verbose")


class LossConfig(Config):
    @property
    def lambda_time_decay(self):
        return self.get_property("lambda_time_decay")

    @property
    def alpha_factor(self):
        return self.get_property("alpha_factor")

    @property
    def gamma(self):
        return self.get_property("gamma")

    @property
    def regression_beta(self):
        return self.get_property("regression_beta")

    @property
    def euler_beta(self):
        return self.get_property("euler_beta")

    @property
    def class_beta(self):
        return self.get_property("class_beta")

    @property
    def confidence_threshold(self):
        return self.get_property("confidence_threshold")

    @property
    def higher_match_threshold(self):
        return self.get_property("higher_match_threshold")

    @property
    def lower_match_threshold(self):
        return self.get_property("lower_match_threshold")

    @property
    def euler_loss(self):
        return self.get_property("euler_loss")

    @property
    def regression_loss(self):
        return self.get_property("regression_loss")


class ModelConfig(Config):
    @property
    def model(self):
        return self.get_property("model")


class TrainConfig(Config):
    @property
    def batch_size(self):
        return self.get_property("batch_size")

    @property
    def shuffle(self):
        return self.get_property("shuffle")

    @property
    def num_workers(self):
        return self.get_property("num_workers")

    @property
    def use_cuda(self):
        return self.get_property("use_cuda")

    @property
    def max_epochs(self):
        return self.get_property("max_epochs")

    @property
    def verbose(self):
        return self.get_property("verbose")

    @property
    def use_visdom(self):
        return self.get_property("use_visdom")

    @property
    def visdom_port(self):
        return self.get_property("visdom_port")

    @property
    def visdom_server(self):
        return self.get_property("visdom_server")

    @property
    def learning_rate(self):
        return float(self.get_property("learning_rate"))

    @property
    def weight_decay(self):
        return float(self.get_property("weight_decay"))

    @property
    def training_seqs(self):
        return self.get_property("training_seqs")

    @property
    def validation_seqs(self):
        return self.get_property("validation_seqs")

    @property
    def cuda_device(self):
        return self.get_property("cuda_device")

    @property
    def multi_gpu(self):
        return self.get_property("multi_gpu")

    @property
    def save_weights_modulus(self):
        return self.get_property("save_weights_modulus")

    @property
    def plot_grad_flow(self):
        return self.get_property("plot_grad_flow")


class EvalConfig(Config):

    @property
    def batch_size(self):
        return self.get_property("batch_size")

    @property
    def num_workers(self):
        return self.get_property("num_workers")

    @property
    def use_cuda(self):
        return self.get_property("use_cuda")

    @property
    def verbose(self):
        return self.get_property("verbose")

    @property
    def use_visdom(self):
        return self.get_property("use_visdom")

    @property
    def cuda_device(self):
        return self.get_property("cuda_device")

    @property
    def confidence_threshold(self):
        return self.get_property("confidence_threshold")

    @property
    def save_figs(self):
        return self.get_property("save_figs")

    @property
    def infer_color(self):
        return tuple(self.get_property("infer_color"))

    @property
    def gt_color(self):
        return tuple(self.get_property("gt_color"))

    @property
    def validation_seqs(self):
        return self.get_property("validation_seqs")

    @property
    def anchors(self):
        return self.get_property("anchors")

    @property
    def num_anchors(self):
        return len(self.get_property("anchors"))

    @property
    def show_confidence(self):
        return self.get_property("show_confidence")

    @property
    def off_screen_rendering(self):
        return self.get_property("off_screen_rendering")

    @property
    def use_gospa(self):
        return self.get_property("use_gospa")

    @property
    def draw_gospa(self):
        return self.get_property("draw_gospa")

    @property
    def save_detections_as_measurements(self):
        return self.get_property("save_detections_as_measurements")

    @property
    def save_raw(self):
        return self.get_property("save_raw")

    @property
    def distance_threshold(self):
        return self.get_property("distance_threshold")


class PostConfig(Config):

    @property
    def iou_threshold(self):
        return self.get_property("iou_threshold")

    @property
    def reason_distance_threshold(self):
        return self.get_property("reason_distance_threshold")

    @property
    def det2pred_distance_threshold(self):
        return self.get_property("det2pred_distance_threshold")

    @property
    def over_time_distance_threshold(self):
        return self.get_property("over_time_distance_threshold")

    @property
    def confidence_threshold(self):
        return self.get_property("confidence_threshold")

    @property
    def verbose(self):
        return self.get_property("verbose")

    @property
    def coordinate_transform(self):
        return self.get_property("coordinate_transform")
