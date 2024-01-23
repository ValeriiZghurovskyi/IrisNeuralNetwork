import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configure logging to display timestamps and log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IrisNet(nn.Module):
    def __init__(self):
        """Define layers of the neural network."""
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        """Define the forward pass of the neural network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_model(model_path):
    """Load the trained model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = IrisNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def softmax(x):
    """Apply the softmax function to turn logits into probabilities."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def run_inference(model, data):
    """Run inference on the given data using the provided model."""
    with torch.no_grad():
        predictions = model(data)
    return predictions

if __name__ == "__main__":
    try:
        # Load settings from the configuration file
        if not os.path.exists('settings.json'):
            raise FileNotFoundError("Settings file not found")

        with open('settings.json') as f:
            settings = json.load(f)

        # Paths for the model and inference data
        model_path = f"{settings['general']['models_dir']}/{settings['inference']['model_name']}"
        inference_data_path = f"{settings['general']['data_dir']}/{settings['inference']['inp_table_name']}"

        # Check if the inference data file exists
        if not os.path.exists(inference_data_path):
            raise FileNotFoundError("Inference data file not found")

        # Load the model and prepare inference data
        model = load_model(model_path)
        inference_data = pd.read_csv(inference_data_path)
        if inference_data.shape[1] < 4:
            raise ValueError("Inference data must have at least 4 columns")

        X_inference = torch.tensor(inference_data.iloc[:, :4].values, dtype=torch.float32)

        # Run inference and calculate the time taken
        start_time = time.time()
        logits = run_inference(model, X_inference).numpy()
        probabilities = softmax(logits)
        predicted_classes = np.argmax(probabilities, axis=1)

        # Save the results to a CSV file
        results = pd.DataFrame({
            'Predicted Class': predicted_classes,
            'Probability Class 0': probabilities[:, 0],
            'Probability Class 1': probabilities[:, 1],
            'Probability Class 2': probabilities[:, 2]
        })

        results_dir = settings['general']['results_dir']
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_path = f"{settings['general']['results_dir']}/inference_results.csv"
        results.to_csv(results_path, index=False)
        end_time = time.time()

        # Log the completion and time taken for inference
        logging.info(f"Inference completed in {end_time - start_time} seconds")

    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")