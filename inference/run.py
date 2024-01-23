"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_project_dir, configure_logging

# Define directories
ROOT_DIR = get_project_dir('')
CONF_FILE = os.path.abspath(os.path.join(ROOT_DIR, 'settings.json'))

# Load configuration settings from JSON
try:
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {CONF_FILE}")
    sys.exit(0)
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON in configuration file: {CONF_FILE}")
    sys.exit(1)


# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])
SCALER_PATH = os.path.join(ROOT_DIR, 'scaler.pickle')

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_scaler(path: str):
    """Loads and returns the saved scaler"""
    try:
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
            logging.info(f'Scaler loaded from {path}')
            return scaler
    except Exception as e:
        logging.error(f'An error occurred while loading the scaler: {e}')
        sys.exit(1)


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str):
    """Loads and returns the specified model"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            logging.info(f'Path of the model: {path}')
            return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, scaler, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""
    model.eval()

    # scaling
    scaled_inputs = scaler.transform(infer_data.values)
    inputs = torch.tensor(scaled_inputs, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)

    _, predicted_classes = torch.max(outputs, 1)
    infer_data['results'] = predicted_classes.numpy()
    return infer_data



def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    scaler = load_scaler(SCALER_PATH)
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))

    scaled_data = scaler.transform(infer_data.values)
    infer_data_scaled = pd.DataFrame(scaled_data, columns=infer_data.columns)

    results = predict_results(model, scaler, infer_data_scaled)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()