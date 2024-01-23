# How to build my project

1. First step. Clone my repository from github, like an example:
git clone https://github.com/ValeriiZghurovskyi/IrisNeuralNetwork.git

2. Second step. Go to IrisNeuralNetwork folder:
cd IrisNeuralNetwork

3. For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image:
```bash
docker build -f ./training/Dockerfile -t training_image .
```

- Run the container to train the model:
```bash
docker run --name <container_name> training_image
```
Replace <container_name> with any name for the container.
This command will start the process of training the model inside the container. It is important to give the container a name so that it is easier to refer to it later.

- Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_name>:/app/models/<model_name>.pkl ./models/
```
Replace <model_name>.pkl with the filename of your model (mentioned in `settings.json`) and <container_name> with the name of your container.

If you don't have `models` folder on your local machine, use next command:
```bash
mkdir -p ./models/
```

2. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```


## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/inference.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -t iris-inference -f inference/Dockerfile .
```

- Run the inference Docker container:
```bash
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results iris-inference
```
Since we have already associated the local `results` folder with the container, the inference results will automatically be stored in this folder on your local machine, so there is no need to use the `docker cp` command.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/inference.py
```

## Tests:
If you need to run tests, you can do so using the command(To run the tests, you need to have the data in the appropriate folder and the model):
```bash
python test_iris.py
```