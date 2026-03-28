FROM python:3.11-slim

WORKDIR /app

# Install system deps for yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy app
COPY callmind/ callmind/

# Create upload dir
RUN mkdir -p uploads

EXPOSE 8000

CMD ["python", "-m", "callmind.app"]
