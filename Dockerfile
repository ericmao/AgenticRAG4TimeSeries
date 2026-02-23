# AgenticRAG4TimeSeries - Layer C demo / CERT pipeline
FROM python:3.11-slim

WORKDIR /app

# So scripts that call python3 work (e.g. run_highrisk_demo.sh)
RUN ln -sf /usr/local/bin/python /usr/local/bin/python3 2>/dev/null || true

# Install dependencies (torch/transformers will increase image size)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (paths in config assume repo root = /app)
COPY . .

# Default: validate (override with docker run/compose run command)
CMD ["python", "-m", "src.cli", "validate"]
