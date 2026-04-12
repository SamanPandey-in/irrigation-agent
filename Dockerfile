FROM python:3.13-slim

LABEL maintainer="Arjun Hackathon Team"
LABEL description="Precision Irrigation RL Agent — OpenEnv Hackathon"

WORKDIR /app

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# System deps for matplotlib headless rendering
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application source
COPY --chown=user . .

# Pre-generate weather data
RUN python scripts/generate_data.py

# Pre-train PPO model so startup is instant (no blocking training at runtime)
RUN mkdir -p models && \
    python scripts/train_and_eval.py --quick --quick-steps 25000 --quick-eval 10 && \
    echo "PPO model trained OK"

# Gradio listens on 7860 (HF Spaces default)
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]