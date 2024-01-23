# Importing required libraries
import pandas as pd
import logging
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import get_project_dir, configure_logging

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

# Define data directory
DATA_DIR = get_project_dir(conf['general']['data_dir'])

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define paths
logger.info("Defining paths...")
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


class IrisDataProcessor:

    def __init__(self, conf, TRAIN_PATH, INFERENCE_PATH):
        self.conf = conf
        self.TRAIN_PATH = TRAIN_PATH
        self.INFERENCE_PATH = INFERENCE_PATH

    def load_iris_data(self):
        logger.info("Loading iris dataset...")
        iris = datasets.load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        target_variable = self.conf['general']['target_variable']
        logger.info(f"Target variable: {target_variable}")
        df[target_variable] = iris.target
        return df

    def split_data(self, df):
        logger.info("Split into training and inference sets...")
        train_data, inference_data = train_test_split(df, test_size=self.conf['general']['inference_size'], random_state=self.conf['general']['random_state'])
        return train_data, inference_data

    def remove_labels(self, inference_data):
        logger.info("Remove labels from inference data...")
        return inference_data.drop(self.conf['general']['target_variable'], axis=1)

    def save_data(self, train_data, inference_data_without_labels):
        logger.info(f"Saving training data to {self.TRAIN_PATH}...")
        train_data.to_csv(self.TRAIN_PATH, index=False)
        logger.info(f"Saving inference data to {self.INFERENCE_PATH}...")
        inference_data_without_labels.to_csv(self.INFERENCE_PATH, index=False)

    def process_and_save_data(self):
        df = self.load_iris_data()
        train_data, inference_data = self.split_data(df)
        inference_data_without_labels = self.remove_labels(inference_data)
        self.save_data(train_data, inference_data_without_labels)


if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    data_processor = IrisDataProcessor(conf, TRAIN_PATH, INFERENCE_PATH)
    data_processor.process_and_save_data()
    logger.info("Script completed successfully.")