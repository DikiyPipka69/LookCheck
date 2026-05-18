FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-util1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]