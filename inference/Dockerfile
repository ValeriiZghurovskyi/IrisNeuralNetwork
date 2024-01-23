# Use Python slim image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code, model files, data, results directory, and settings to the container
COPY inference/inference.py .
COPY models/ ./models
COPY data/ ./data
COPY settings.json .

# Run the inference script when the container launches
CMD ["python", "inference.py"]