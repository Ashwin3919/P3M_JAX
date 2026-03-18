import json
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract config name from filename for result storage
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    config['name'] = config_name
    
    return config

def get_results_dir(config_name):
    results_dir = os.path.join('results', config_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
