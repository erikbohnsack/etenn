import yaml
import os


def get_root_dir():
    if os.path.exists("/Users/erikbohnsack/data/"):
        root_dir = "/Users/erikbohnsack/data/"
    elif os.path.exists('/home/mlt/adam/fafe/cfg/ai_data'):
        root_dir = "/home/mlt/adam/data/"
    elif os.path.exists('/home/mlt/data/'):
        root_dir = "/home/mlt/data/"
    elif os.path.exists('/mnt/home/a315255/data/'):
        root_dir = "/mnt/home/a315255/data/"
    else:
        raise ValueError("No data folders found.")
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    return root_dir


def get_showroom_path(model_path, full_path_bool):
    if full_path_bool:
        showroom_path = os.path.join('showroom', model_path.split('/')[-1])
    else:
        showroom_path = os.path.join('showroom', model_path)

    if not os.path.exists('showroom'):
        os.mkdir('showroom')

    if not os.path.exists(showroom_path):
        os.mkdir(showroom_path)
        os.mkdir(os.path.join(showroom_path, 'oracle_view'))
        os.mkdir(os.path.join(showroom_path, 'top_view'))
        os.mkdir(os.path.join(showroom_path, 'grad_flow_pillar'))
        os.mkdir(os.path.join(showroom_path, 'grad_flow_fafe'))

    return showroom_path


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(config_file, data):
    with open(config_file, 'w') as stream:
        yaml.dump(data, stream, default_flow_style=False)
