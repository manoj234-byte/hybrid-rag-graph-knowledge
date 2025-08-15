FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    NLTK_DATA=/app/.cache/nltk_data \
    HF_HOME=/app/.cache \
    HF_DATASETS_CACHE=/app/.cache \
    XDG_CACHE_HOME=/app/.cache

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create necessary cache and config dirs
RUN mkdir -p /app/.cache /app/.streamlit

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data including punkt and punkt_tab
RUN python -m nltk.downloader -d /app/.cache/nltk_data punkt punkt_tab

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app source
COPY . .

# Add a non-root user and switch
RUN useradd -m appuser && \
    chown -R appuser /app
USER appuser

# Streamlit config (avoids config warnings)
RUN echo "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /app/.streamlit/config.toml

# Expose the port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "src/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]
