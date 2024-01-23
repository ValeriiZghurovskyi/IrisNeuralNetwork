# Using the Python slim image as the base image
FROM python:3.9-slim

# Setting the working directory in the container
WORKDIR /app

# Copying all necessary files into the container
COPY training/ ./training
COPY data/ ./data
COPY settings.json .

# Copying the requirements file into the container
COPY requirements.txt .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Running the training script
CMD ["python", "./training/train.py"]
