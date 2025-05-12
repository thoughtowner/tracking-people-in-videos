FROM python:3.12-slim

# Устанавливаем необходимые библиотеки для сборки (в том числе cmake и другие инструменты)
RUN apt-get update && \
    apt-get install -y \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-dev \
    libx264-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip, setuptools и wheel до последних версий
RUN pip install --upgrade pip setuptools wheel

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код в контейнер
COPY . /code

# Устанавливаем рабочую директорию
WORKDIR /code

# Запуск приложения с uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
