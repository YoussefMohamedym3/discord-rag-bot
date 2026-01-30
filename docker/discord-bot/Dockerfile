# Use a slim python image to keep size down
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY src/ ./src
COPY data/ ./data

# Copy entrypoint script
COPY docker/discord-bot/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set python path so it can find your modules in src
ENV PYTHONPATH=/app/src

ENTRYPOINT ["entrypoint.sh"]