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

# Copy dependency files first (layer cache optimisation)
COPY --chown=user requirements.txt pyproject.toml ./

# Install all Python dependencies
# numpy 2.x is required — numpy 1.x has no Python 3.13 wheel and triggers
# a Meson/C-compiler build that fails in this slim image.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY --chown=user . .

# Install the package itself so [project.scripts] entry points are registered
# and `openenv validate` can resolve `server.app:main`
RUN pip install --no-cache-dir -e . --no-deps

# Pre-generate weather data
RUN python scripts/generate_data.py

# Create stub PPO model so server starts instantly; app.py retrains in background
RUN python scripts/create_stub_model.py

# Gradio listens on 7860 (HF Spaces default)
EXPOSE 7860

CMD ["python", "app.py"]
