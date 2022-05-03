import argparse
import yaml
import os
import shutil

from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config_files/config_train.yaml',
                        type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    if not os.path.isdir("{}/{}/".format(cfg['output']['output_folder'], cfg['output']['description'])):
        os.makedirs("{}/{}/".format(cfg['output']['output_folder'], cfg['output']['description']))
    shutil.copy(args.cfg, "{}/{}/".format(cfg['output']['output_folder'], cfg['output']['description']))

    trainer = Trainer(cfg)
    trainer.train()