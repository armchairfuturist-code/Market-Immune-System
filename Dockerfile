FROM python:3.10-slim

# Install curl for health checks (if needed)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run uses 8080 by default
EXPOSE 8080

# Update healthcheck to look at the correct port (8080)
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# These flags are CRITICAL for Firebase Hosting + Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.enableWebsocketCompression=false"]
