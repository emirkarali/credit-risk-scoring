FROM python:3.11-slim

WORKDIR /app

# Sistem gereksinimleri (LightGBM için libgomp gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# FastAPI portu
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]