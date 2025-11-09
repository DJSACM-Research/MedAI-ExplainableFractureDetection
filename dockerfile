FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application
COPY . .

# Create outputs directory for models
RUN mkdir -p outputs

# Expose port (default 8501, can be overridden by PORT env var)
EXPOSE 8501

# Run Streamlit with proper configuration for production
CMD streamlit run streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info
