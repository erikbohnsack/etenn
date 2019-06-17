import torch
from torchviz import make_dot
from models.fafenet import FafeNet
from cfg.config import InputConfig, TrainConfig
from cfg.config_stuff import load_config, get_root_dir

def create_FafeNet_graph():
    config_path = '../cfg/cfg.yml'
    config = load_config(config_path)
    input_config = InputConfig(config['INPUT_CONFIG'])
    train_config = TrainConfig(config['TRAIN_CONFIG'])

    root_dir = get_root_dir()
    model = FafeNet(input_config=input_config)
    x = torch.randn(1, 80, 350, 400).requires_grad_(True)
    y = model(x)
    dot = make_dot(y, params={**{'inputs': x}, **dict(model.named_parameters())})
    dot.format = 'pdf'
    dot.render('../showroom/FafeNet')

if __name__ == "__main__":
    create_FafeNet_graph()
