import unittest
import os
import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from training.train import IrisNet, train_model
from data_process.data_generation import generate_iris_data
from inference.inference import run_inference, load_model

class TestIrisData(unittest.TestCase):
    """Test cases for the Iris data generation and splitting functionality."""

    def setUp(self):
        """Load settings and generate data before each test."""
        with open('settings.json') as f:
            self.settings = json.load(f)
        generate_iris_data(self.settings)

    def test_data_split(self):
        """Ensure that the data is correctly split into training and inference sets."""
        data_dir = self.settings['general']['data_dir']
        train_path = os.path.join(data_dir, self.settings['train']['table_name'])
        inference_path = os.path.join(data_dir, self.settings['inference']['inp_table_name'])
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(inference_path))

class TestIrisModel(unittest.TestCase):
    """Test cases for the Iris model training and inference."""

    def setUp(self):
        """Load settings and initialize the model before each test."""
        with open('settings.json') as f:
            self.settings = json.load(f)
        self.model = IrisNet()

    def test_model_can_train(self):
        """Check if the model can be trained with the provided settings."""
        self.assertIsNone(train_model(self.settings))

    def test_model_saving_loading(self):
        """Ensure that the model can be saved and then loaded."""
        model_path = os.path.join(self.settings['general']['models_dir'], 'temp_model.pkl')
        torch.save(self.model.state_dict(), model_path)
        self.assertTrue(os.path.exists(model_path))
        loaded_model = IrisNet()
        loaded_model.load_state_dict(torch.load(model_path))
        os.remove(model_path)  # Clean up the temporary model file

    def test_inference(self):
        """Verify that the inference process produces predictions."""
        model_path = os.path.join(self.settings['general']['models_dir'], self.settings['inference']['model_name'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        data_dir = self.settings['general']['data_dir']
        inference_data_path = os.path.join(data_dir, self.settings['inference']['inp_table_name'])
        inference_data = pd.read_csv(inference_data_path)
        X_inference = torch.tensor(inference_data.iloc[:, :4].values, dtype=torch.float32)
        predictions = run_inference(self.model, X_inference)
        self.assertEqual(predictions.shape[0], X_inference.shape[0])

if __name__ == '__main__':
    unittest.main()
