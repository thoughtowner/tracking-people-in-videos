FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libx11-dev libx264-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /code

WORKDIR /code

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
