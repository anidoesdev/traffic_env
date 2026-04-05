# Smart City Traffic Flow — OpenEnv Docker Image --------------─
#
# Hugging Face Spaces requires port 7860.
# TASK_LEVEL controls which task runs (easy / medium / hard).
#
# Build:  docker build -t traffic-env .
# Run:    docker run -p 7860:7860 -e TASK_LEVEL=easy traffic-env
# Test:   curl http://localhost:7860/health

FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Set PYTHONPATH so bare imports work inside Docker (e.g. "from models import ...")
ENV PYTHONPATH=/app

# Task level — override with -e TASK_LEVEL=hard at runtime
ENV TASK_LEVEL=easy

# Hugging Face Spaces always uses port 7860
EXPOSE 7860

# Health check — openenv validate uses this to confirm the server is up
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
