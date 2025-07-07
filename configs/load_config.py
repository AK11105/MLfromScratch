import yaml
import os

def load_config(path='configs/defaults.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
