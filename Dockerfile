# Используем версию Bookworm (Debian 12), с ней меньше проблем у Хрома
FROM python:3.11-slim-bookworm

# 1. Устанавливаем Chromium и драйвер
# Это самая важная часть. Без этого Selenium скажет "Browser not found"
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Указываем переменные, чтобы Selenium знал, где искать браузер
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Настройки Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Ставим библиотеки
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Копируем код
COPY . .

# Создаем папку для отчетов/данных
RUN mkdir -p reports data

# Запускаем (замени на свой файл запуска)
CMD ["python", "main.py"]
