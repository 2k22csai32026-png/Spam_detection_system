# Use an official lightweight Python image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts, application code, and UI
COPY model_files/ ./model_files/
COPY main.py .
COPY index.html .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 so external traffic reaches the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]