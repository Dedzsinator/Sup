FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 spam_detector && chown -R spam_detector:spam_detector /app
USER spam_detector

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"]
