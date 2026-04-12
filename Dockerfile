FROM python:3.12-slim

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

COPY --chown=user requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN pip install --no-cache-dir -e . --no-deps

RUN python scripts/generate_data.py
RUN python scripts/create_stub_model.py

EXPOSE 7860
CMD ["python", "app.py"]