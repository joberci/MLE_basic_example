# MLE Basic Example

This repository contains a basic example of a Machine Learning Engineering (MLE) project structured for training and inference using Docker. The project involves training a simple neural network model on the Iris dataset and making predictions with the trained model.

## Project Structure

The project structure is organized as follows:

```
MLE_basic_example
├── data                      # datafiles for training and inference in csv format
│   ├── iris_inference_data.csv
│   └── iris_train_data.csv
├── data_process              # data processing script that loads and splits the Iris dataset 
│   ├── data_processing.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
├── Joni_Bertalan_HW_08_Iris.ipynb  # Jupyter notebook
└── README.md
```

## Prerequisites

In order to run the project, you only need the Docker application installed on your computer. Alternatively, you can work directly on your local machine using Python and Git. All the required packages are listed in the `requirements.txt` file. 


## Forking and Cloning from GitHub
To begin using this project, you should initiate the process by creating your own copy on GitHub through the 'forking' mechanism. Navigate to the main page of the `ML_basic_example` project, and click on the 'Fork' button located at the top right corner. This action will duplicate the project under your personal GitHub account. Subsequently, you can 'clone' the repository to your local machine for individual use. To achieve this, click the 'Code' button on your forked repository, copy the provided link, and execute the `git clone` command in your terminal using the copied link. This procedure establishes a local duplicate of the repository on your machine, allowing you to start working on your machine.


## Settings:
The `settings.json` file serves as the central configuration hub for this data science project, encapsulating various parameters and options that govern its behavior. Below is a breakdown of the key sections and their respective functionalities:

### General Configuration:
- **`random_state`**: An integer specifying the random seed for reproducibility.
- **`datetime_format`**: A string defining the format for displaying date and time in logs and file names.
- **`data_dir`**: The directory where the project's data files are stored.
- **`models_dir`**: The directory housing the trained model files.
- **`results_dir`**: The directory for storing the results of model predictions.
- **`target_variable`**: The target variable used in the dataset.
- **`inference_size`**: A float representing the fraction of the dataset reserved for inference.

### Training Configuration:
- **`table_name`**: The filename of the training dataset.
- **`test_size`**: The proportion of the dataset reserved for testing during model training.
- **`batch_size`**: The batch size used in training the machine learning model.
- **`epochs`**: The number of training epochs.
- **`learning_rate`**: The learning rate for the optimizer during model training.
- **`verbose`**: An integer specifying how frequently to print training loss information.
- **`plot`**: An integer indicating whether to plot the learning curve during training (0 for no plot, 1 for plot).

### Inference Configuration:
- **`inp_table_name`**: The filename of the dataset used for model inference.
- **`model_name`**: The filename of the trained model used for inference.

Adjust these settings as needed to tailor the project to your specific requirements. The `settings.json` file consolidates these configuration parameters, providing a centralized and easily modifiable control panel for your data science workflows.

## Data Exploration
You can find detailed exploratory data analysis in `Joni_Bertalan_HW_08_Iris.ipynb` file.
The data processing and model building used in this project were analyzed and determined in this notebook. 

## Data Directory: `data/`

The `data/` directory is a repository for the essential datasets used in this project. It contains two CSV files.

### Dataset Files:

1. **`iris_train_data.csv`**: This CSV file comprises the training dataset used to train the machine learning model. It includes features and corresponding labels required for the model to learn patterns and relationships.

2. **`iris_inference_data.csv`**: This CSV file houses the dataset reserved for model inference. It consists of feature information only, allowing the trained model to make predictions based on unseen data.

### Data Processing:
Both datasets were generated and processed using the `data_processing.py` script. 
These datasets are ready for use, and no further interventions are necessary. 
If needed, you can execute this script to regenerate or modify the datasets based on your specific requirements.


## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.


Data preparation strategy in the data processing step:

1. **Drop Duplicates**: The data cleaning step only includes the "drop duplicate" part this time, therefore we perform it before train-test split.
2. **Train-Test Split**: The next step is the train-test split, which is good practice to avoid data leakage. The code correctly applies the train_test_split function. As default, we split the data in such a manner that 70% of the data will be in the train dataset, and the rest 30% will be the test dataset (but this can be adjusted in the settings file).
3. **Scaling (MinMaxScaler)**: Scaling is performed using the MinMaxScaler, and it is applied to both, train and test datasets. Importantly, the learned parameters from the training dataset are used for standardization on the test dataset as well. Scaling is also saved into `scaler.pickle` file to be able to use the same scaling for inference data later. 
4. **Creating PyTorch DataLoaders:** Finally, PyTorch DataLoaders are created for the training and test datasets.

Based on this, the data processing function correctly implements the train-test split and scaling method without encountering data leakage issues. The code is well-structured and easy to understand.

The model used in the training is the final best model developed and discussed in the `Joni_Bertalan_HW_08_Iris.ipynb` file.




### To train the model using Docker: 

- Build the training Docker image using the following command:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You can run the container using the following command:
```bash
docker run -it training_image
```
After running the container, copy the trained model and the saved scaler file from Docker container to your local machine using the following commands: 

```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
docker cp <container_id>:/app/scaler.pickle .
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

Alternatively, the `train.py` script can also be run locally.



## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

### To run the inference using Docker: 
To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pickle --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -it inference_image
```
- You can copy the results from the Docker container to your local `results` directory using the following command:
```bash
docker cp <container_id>:/app/results/<result_name>.csv ./results
```

Alternatively, you can also run the inference script locally.

