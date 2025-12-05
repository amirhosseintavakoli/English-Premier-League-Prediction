
# Dockerfile
FROM python:3.11-slim

# 1) System basics (only if needed for common wheels like lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 2) Create app directory
WORKDIR /app

# 3) Install Python deps with caching
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4) Copy source
COPY . /app

# 5) Security: run as non-root user
RUN useradd -m appuser
USER appuser

# 6) Default port variable (many platforms inject $PORT)
ENV PORT=8501

# 7) Healthcheck (Streamlit exposes a health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsSL "http://localhost:${PORT}/_stcore/health" || exit 1

# 8) Expose for local convenience (not required for all PaaS)
EXPOSE 8501

# 9) Start Streamlit: bind to 0.0.0.0 and respect $PORT
CMD streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT}
