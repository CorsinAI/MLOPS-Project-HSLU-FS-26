FROM python:3.11-slim

WORKDIR /app

# gcc is required to compile twofish (a hopsworks dependency)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command for HuggingFace Spaces (port 7860 required)
# Docker Compose overrides this with its own command for local use
CMD ["python", "-m", "pipelines.inference.run", "--host", "0.0.0.0", "--port", "7860"]
