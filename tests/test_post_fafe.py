import unittest
import torch
from post_fafe import PostFafe
from cfg.config import InputConfig, PostConfig
from cfg.config_stuff import load_config
import os


class TestPostFafe(unittest.TestCase):
    def test_data_structure(self):
        if os.path.exists('/home/mlt/mot/fafe/cfg/adams_computer'):
            config_path = 'cfg/cfg_pp_mini.yml'
        elif os.path.exists('/Users/erikbohnsack'):
            config_path = 'cfg/cfg_mac.yml'
        else:
            config_path = 'cfg/cfg_pp.yml'

        print('Using config: \n\t{}\n'.format(config_path))
        config = load_config(config_path)
        input_config = InputConfig(config['INPUT_CONFIG'])
        post_config = PostConfig(config['POST_CONFIG'])
        data = torch.load('inference0.pt')
        data1 = torch.load('inference1.pt')
        data2 = torch.load('inference2.pt')

        device = torch.device("cpu")

        post_fafe = PostFafe(input_config, post_config, device)
        print(post_fafe.object_state)
        post_fafe(data[0])
        print(post_fafe.object_state.keys())
        post_fafe(data1[0])
        print(post_fafe.object_state.keys())
        post_fafe(data2[0])
        print(post_fafe.object_state.keys())
