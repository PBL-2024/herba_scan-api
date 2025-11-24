# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirement.txt .

# Install any needed packages specified in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
