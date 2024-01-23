import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_iris_data(settings):
    """
    Generate and split Iris dataset into training and inference sets,
    and save them as CSV files.
    """
    try:
        data_dir = settings['general']['data_dir']
        train_config = settings['train']
        inference_config = settings['inference']

        # Check if data directory exists, create if not
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Load Iris dataset
        # i will load it from sklearn.datasets, Because it was written on Wikipedia that the iris data set
        # is widely used as a beginner's dataset for machine learning purposes. The dataset is included in
        # R base and Python in the machine learning library scikit-learn, so that users can access it without
        # having to find a source for it.
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Split the dataset into training and inference sets
        train, inference = train_test_split(df, test_size=train_config['test_size'], random_state=settings['general']['random_state'])

        # Save the datasets to CSV files
        train.to_csv(f'{data_dir}/{train_config["table_name"]}', index=False)
        inference.to_csv(f'{data_dir}/{inference_config["inp_table_name"]}', index=False)

        logging.info("Iris data successfully generated and saved.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        with open('settings.json') as f:
            settings = json.load(f)
        generate_iris_data(settings)
    except FileNotFoundError:
        logging.error("The settings.json file was not found.")
    except json.JSONDecodeError:
        logging.error("Error decoding settings.json.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
