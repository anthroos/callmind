FROM python:3.11-slim

WORKDIR /app

# Install system deps for yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python deps directly
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.32.0" \
    jinja2>=3.1.0 \
    python-multipart>=0.0.12 \
    google-genai>=1.0.0 \
    yt-dlp>=2024.12.0 \
    httpx>=0.27.0 \
    qdrant-client>=1.12.0 \
    fastembed>=0.4.0 \
    python-dotenv>=1.0.0

# Copy app
COPY callmind/ callmind/

# Create upload dir
RUN mkdir -p uploads

EXPOSE 8000

CMD ["python", "-m", "callmind.app"]
