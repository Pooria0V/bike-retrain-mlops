# Dockerfile

# --- Base image ---
# python:3.12-slim is a minimal Python image without unnecessary system packages.
# "slim" keeps the image size small — important for faster deployment.
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# --- Install dependencies first (before copying source code) ---
# Why copy requirements.txt separately before copying the rest?
# Docker builds images in layers. If we copy everything at once,
# any code change would invalidate the requirements layer and
# force a full re-install. This way, dependencies are only
# re-installed when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<81" && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy source code only ---
# Data is mounted at runtime via docker-compose volumes,
# not baked into the image — keeping the image small and portable.
COPY src/ ./src/

# Expose the port FastAPI will run on
EXPOSE 8000

# --- Start the API ---
# --host 0.0.0.0 makes the server accessible from outside the container
# --workers 2 runs two parallel worker processes for better throughput
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]