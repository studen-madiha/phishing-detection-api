# Base image for Python
FROM python:3.9

# Working directory
WORKDIR /app

# Requirements install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Saari files copy karein
COPY . .

# Flask app ko run karein
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]