FROM python:3.11-slim

WORKDIR /app

# gcc is required to compile twofish (a hopsworks dependency)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
