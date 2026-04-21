FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the FastAPI port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
