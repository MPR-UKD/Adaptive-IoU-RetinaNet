# Basis-Image
FROM python:3.9-slim-buster

# Arbeitsverzeichnis im Container
WORKDIR /app

# Kopieren der erforderlichen Dateien in das Arbeitsverzeichnis
COPY . .

# Installation von Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Verwenden von Python aus venv
ENV PATH="/app/venv/bin:$PATH"