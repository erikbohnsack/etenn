import unittest
import torch
from eval import eval, eval_with_GT
from models.fafenet import FafeNet
from train import InputConfig
from cfg.config_stuff import load_config


class TestEval(unittest.TestCase):
    def test_eval(self):
        save_filename = 'foo'
        config_path = 'cfg.yml'
        config = load_config(config_path)

        input_config = InputConfig(config['INPUT_CONFIG'])
        model = FafeNet(config)
        torch.save({'model_state_dict': model.state_dict()}, save_filename)

        model_path = save_filename
        data_path = '/Users/erikbohnsack/data'
        eval(model_path=model_path, data_path=data_path)

    def test_eval_with_model(self):
        model_path = '../test_train_2019-04-10_16_59_epoch_48'
        data_path = '/Users/erikbohnsack/data'
        config_path = '../cfg/cfg.yml'
        eval_with_GT(model_path=model_path, data_path=data_path, config_path=config_path)


if __name__ == 'main':
    TestEval.test_eval_with_model()
