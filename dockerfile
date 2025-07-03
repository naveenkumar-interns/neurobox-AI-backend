# Use official Python runtime as the base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies (if needed, e.g., for pymongo or web scraping)
RUN apt-get update && apt-get install -y --no-install-recommends \

# Copy requirements file first (optimization for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables (optional defaults, override with docker run -e)
ENV PYTHONUNBUFFERED=1
ENV USER_AGENT="JacksonHardwareBot/1.0"

# Expose port 5000
EXPOSE 5000

# Command to run the application with Gunicorn
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]