apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt