# Use an ARM-compatible base image with Python and pip
FROM python:3.10-slim

# Install system dependencies required for MMAction2 and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Upgrade pip and install pipreqs
RUN pip install --upgrade pip && \
    pip install pipreqs

# Generate requirements.txt based on actual imports in the project
RUN pipreqs . --force

# Install only the required Python dependencies
RUN pip install -r requirements.txt

# Optional: Set default command to run your app (e.g., your inference script)
CMD ["python", "localizer.py"]
