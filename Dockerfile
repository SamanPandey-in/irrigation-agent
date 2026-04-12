FROM python:3.13-slim

LABEL maintainer="Arjun Hackathon Team"
LABEL description="Precision Irrigation RL Agent — OpenEnv Hackathon"

WORKDIR /app

# System deps first, as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application source
COPY --chown=user . .

# Pre-generate weather data
RUN python scripts/generate_data.py

# Create stub PPO model so server starts instantly
RUN python scripts/create_stub_model.py

# Gradio listens on 7860 (HF Spaces default)
EXPOSE 7860

CMD ["python", "app.py"]