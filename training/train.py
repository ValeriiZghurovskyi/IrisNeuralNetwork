import os
import json
import logging
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IrisNet(nn.Module):
    def __init__(self):
        """Initialize layers for the IrisNet neural network."""
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
        return self.fc4(x)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance on the test set."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
    return accuracy

def train_model(settings):
    """Train the IrisNet model with given settings."""
    try:
        data_dir = settings['general']['data_dir']
        train_path = f"{data_dir}/{settings['train']['table_name']}"
        models_dir = settings['general']['models_dir']
        model_path = f"{models_dir}/{settings['inference']['model_name']}"

        logging.info("Loading and preparing data...")
        data = pd.read_csv(train_path)
        X = data.drop('target', axis=1).values
        y = LabelEncoder().fit_transform(data['target'])

        # Splitting the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings['train']['test_size'], random_state=settings['general']['random_state'])

        # Converting data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

        model = IrisNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        logging.info(f"Size of training dataset: {len(X_train)}")
        logging.info(f"Size of test dataset: {len(X_test)}")

        logging.info("Starting training...")
        start_time = time.time()

        # Training process
        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            accuracy = evaluate_model(model, X_test, y_test)
            logging.info(f"Epoch {epoch+1}/{50}, Loss: {loss.item()}, Accuracy: {accuracy}")

        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds")

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        torch.save(model.state_dict(), model_path)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    try:
        with open('settings.json') as f:
            settings = json.load(f)
        train_model(settings)
    except FileNotFoundError:
        logging.error("The settings.json file was not found.")
    except json.JSONDecodeError:
        logging.error("Error decoding settings.json.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")