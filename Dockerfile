FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages system-wide (removed --user)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002

# Explicit path to uvicorn (safety measure)
CMD ["/usr/local/bin/uvicorn", "chatbot_openai:app", "--host", "0.0.0.0", "--port", "5002", "--workers", "1"]