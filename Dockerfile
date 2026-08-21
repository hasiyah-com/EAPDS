FROM python:3.10

WORKDIR /app

# --- โค้ดและ config ---
COPY app.py ./
COPY firebase_key.json ./
COPY requirements.txt ./

# --- โมเดลที่ระบบใช้งานจริง 3 ตัว ---
COPY models/deberta_finetuned_v579 ./models/deberta_finetuned_v579
COPY model_full/url_full ./model_full/url_full
COPY models_emotion45 ./models_emotion45

RUN apt-get update && apt-get install -y \
    build-essential python3-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn transformers scikit-learn pandas joblib tensorflow==2.15.0
RUN pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
