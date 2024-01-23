"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import matplotlib.pyplot as plt

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
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
SCALER_PATH = os.path.join(ROOT_DIR, 'scaler.pickle')

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, path: str):
        """
        Data processing for reading, preprocessing, and preparing PyTorch dataloaders.
        """
        logging.info(f"Preparing data for training from {path}...")
        start_time = datetime.now()

        # load data
        target_variable = conf['general']['target_variable']
        df = self.data_extraction(path)

        # extract feature names
        features = df.columns.tolist()
        features.remove(target_variable)
        logging.info(f"features: {features}")

        # clean data (drop duplicates)
        df = self.drop_duplicates(df)

        # train-test split (before any data preparation in order to avoid data leakage)
        train, test = self.data_split(df)
        logging.info(f"train dataset: {len(train)} ({round(100*len(train)/len(df),1)} %)")
        logging.info(f"test  dataset: {len(test)} ({round(100*len(test)/len(df),1)} %)")

        # Scaling
        train_scaled, test_scaled = self.data_scaling(train, test, features)

        # Separation of the target variable from the predictior features
        logging.info("Separation of the target variable from the predictior features...")
        X_train, y_train = train_scaled.drop(target_variable, axis=1), train_scaled[target_variable]
        X_test, y_test = test_scaled.drop(target_variable, axis=1), test_scaled[target_variable]

        # Convert to PyTorch tensors
        logging.info("Convert to PyTorch tensors...")
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        # Create DataLoader
        logging.info("Create DataLoader...")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=conf['train']['batch_size'], shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=conf['train']['batch_size'], shuffle=False)

        end_time = datetime.now()
        processing_time = end_time - start_time
        logging.info(f"Data preparation completed in {processing_time}.")

        return train_loader, test_loader

    def data_extraction(self, path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.
        """
        try:
            logging.info(f"Loading data from {path}...")
            df = pd.read_csv(path)
            logging.info("Data loaded successfully.")
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading data from {path}: {str(e)}")
            raise

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicate rows from the DataFrame.
        """
        logging.info(f"Dropping duplicates: {(df.duplicated() == True).sum()}")
        return df.drop_duplicates()

    def data_split(self, df: pd.DataFrame):
        """
        train-test split (before any data preparation in order to avoid data leakage)
        """
        logging.info("Splitting data into training and test sets...")
        return train_test_split(df, test_size=conf['train']['test_size'],
                                random_state=conf['general']['random_state'])

    def data_scaling(self, train: pd.DataFrame, test: pd.DataFrame, features: list):
        logging.info("Scaling...")
        scaler = MinMaxScaler()
        train_scaled = copy.deepcopy(train)
        test_scaled = copy.deepcopy(test)
        train_scaled[features] = scaler.fit_transform(train_scaled[features])
        test_scaled[features] = scaler.transform(test_scaled[features])

        with open(SCALER_PATH, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)

        return train_scaled, test_scaled

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


def train_and_evaluate_model(model, train_dataloader, validation_dataloader, epochs, lr, model_name='', verbose=0, plot=False):
    """
    Train and evaluate a PyTorch model using the specified training and validation dataloaders.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained and evaluated.
    - train_dataloader (DataLoader): Dataloader for the training dataset.
    - validation_dataloader (DataLoader): Dataloader for the validation dataset.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate for the optimizer.
    - model_name (str): Name or description of the model, used in the learning curve plot.
    - verbose (int, optional): If greater than 0, print the average training loss every 'verbose' epochs. Default is 0.

    Returns:
    - model (nn.Module): Trained PyTorch model.

    The function performs model training using the specified optimizer and loss function (CrossEntropyLoss for classification).
    It prints the average training loss at specified intervals and plots the learning curve.
    The model is trained for the specified number of epochs and returned.
    """

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    # CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        train_losses.append(average_loss)

        if verbose and epoch % verbose == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}')

        # Validation loss
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        average_val_loss = val_loss / len(validation_dataloader)
        val_losses.append(average_val_loss)

    # Plotting the learning curve
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curve {model_name}')
        plt.legend()
        plt.ylim(0, max(max(train_losses), 1.2 * max(val_losses)))
        plt.show()

    return model


def save_model(model, path=None):
    logging.info("Saving the model...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not path:
        path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
    else:
        path = os.path.join(MODEL_DIR, path)

    with open(path, 'wb') as f:
        pickle.dump(model, f)


def evaluate_model(model, validation_dataloader):
    """
    Evaluate a PyTorch model on the validation dataset.

    Parameters:
    - model (nn.Module): The PyTorch model to be evaluated.
    - validation_dataloader (DataLoader): Dataloader for the validation dataset.

    Prints:
    - Validation accuracy.
    - Classification report.

    The function evaluates the model on the validation dataset, calculating accuracy and printing the classification report.
    """

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in validation_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # Use torch.max to get the predicted class indices

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Print the classification report
    logging.info(f"Classification Report:\n{classification_report(all_targets, all_predictions)}")


def main():
    configure_logging()

    torch.manual_seed(conf['general']['random_state'])

    data_proc = DataProcessor()
    train_dataloader, validation_dataloader = data_proc.prepare_data(TRAIN_PATH)

    simple_model = SimpleNN()
    logging.info(f'Simple Neural Network Model:\n{simple_model}')

    logging.info(f'Training model...')
    simple_trained_model = train_and_evaluate_model(
        simple_model,
        train_dataloader,
        validation_dataloader,
        epochs=conf['train']['epochs'],
        lr=conf['train']['learning_rate'],
        verbose=conf['train']['verbose'],
        plot=bool(conf['train']['plot'])
    )
    logging.info(f'Model training completed.')

    logging.info(f'Model evaluation...')
    evaluate_model(simple_trained_model, validation_dataloader)

    save_model(simple_trained_model)
    logger.info("Script completed successfully.")

if __name__ == "__main__":
    main()