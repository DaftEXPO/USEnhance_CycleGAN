from pipeline import *

import argparse
import os
import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    trainer = Cyc_Trainer(config)
    trainer.inference()


###################################
if __name__ == '__main__':
    main()