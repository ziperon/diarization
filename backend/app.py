import os
import time
import json
import tempfile
import logging
import subprocess
import smtplib
from email.mime.text import MIMEText
import asyncio
import gc
import librosa
import soundfile as sf
from fastapi import FastAPI
import boto3
import torch
import torchaudio

import numpy as np
from pyannote.audio import Pipeline
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline as hf_pipeline
)
from huggingface_hub import snapshot_download

# ------------------- Конфиги -------------------
S3_BUCKET = "dionrecord"
LOCAL_TMP = "/tmp/audiot"
CHECK_INTERVAL = 10
SUPPORTED_EXT = ['mp3', 'm4a', 'wav', 'flac']

EMAIL_HOST = "smtp.mailmug.net"
EMAIL_PORT = 2525
EMAIL_USER = "rv52j9uijrxg83fv"
EMAIL_PASS = "i2qukytuj2hrtunr"
EMAIL_TO = "your-email@gmail.com"

MODELS_DIR = "./models"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Вернули small для скорости

os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------- Логирование -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------- S3 -------------------
s3 = boto3.client(
    "s3",
    endpoint_url='https://10.76.50.8:9000',
    aws_access_key_id='ntA4tkufij5GsOZfDpNf',
    aws_secret_access_key='zOCaL96ZdlECPy2rU5Pz4ffbPfvFWlYD3bSdrTGt',
    verify=False
)

# ------------------- FastAPI -------------------
app = FastAPI()

# ------------------- Оптимизация скорости -------------------
class PerformanceOptimizer:
    """Оптимизатор производительности"""
    
    @staticmethod
    def get_available_device():
        """Определяем лучшее доступное устройство"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 4:  # Если GPU с 4+ GB памяти
                logging.info(f"🎯 Используем GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
                return "cuda"
        
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            logging.info("🎯 Используем Apple MPS")
            return "mps"
        
        logging.info("🎯 Используем CPU")
        return "cpu"
    
    @staticmethod
    def optimize_torch():
        """Оптимизируем PyTorch для скорости"""
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def log_processing_time(start_time, operation_name):
        """Логируем время выполнения операции"""
        elapsed = time.time() - start_time
        logging.info(f"⏱ {operation_name}: {elapsed:.1f} сек")

# ------------------- Быстрая конвертация -------------------
def convert_to_wav_fast(input_path):
    """Быстрая конвертация аудио"""
    logging.info(f"⚡ Конвертируем {os.path.basename(input_path)}...")
    start_time = time.time()
    
    temp_path = input_path.rsplit('.', 1)[0] + "_fast.wav"
    
    try:
        # Быстрая конвертация с оптимизированными параметрами
        cmd = [
            "ffmpeg", "-y", 
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "-threads", "2",  # Ограничиваем потоки для стабильности
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        os.replace(temp_path, input_path)
        PerformanceOptimizer.log_processing_time(start_time, "Конвертация")
        return input_path
        
    except Exception as e:
        logging.error(f"❌ Ошибка конвертации: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# ------------------- Быстрая загрузка моделей -------------------
def load_pyannote_fast():
    """Быстрая загрузка pyannote"""
    logging.info("⚡ Загружаем pyannote...")
    start_time = time.time()
    
    try:
        hf_token = "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"
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            token=hf_token,
            cache_dir=MODELS_DIR
        )
        
        device = PerformanceOptimizer.get_available_device()
        pipeline = pipeline.to(torch.device(device))
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка PyAnnote")
        return pipeline
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки pyannote: {e}")
        raise

def load_whisper_fast():
    """Быстрая загрузка Whisper"""
    logging.info("⚡ Загружаем Whisper...")
    start_time = time.time()
    
    try:
        local_path = os.path.join(MODELS_DIR, "whisper-trubo")
        
        if not os.path.exists(local_path):
            logging.info("📥 Скачиваем модель...")
            local_path = snapshot_download(repo_id=WHISPER_MODEL, cache_dir=MODELS_DIR)

        processor = WhisperProcessor.from_pretrained(local_path)
        model = WhisperForConditionalGeneration.from_pretrained(local_path)
        
        device = PerformanceOptimizer.get_available_device()
        model.to(device)
        model.eval()
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка Whisper")
        return processor, model, device
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки Whisper: {e}")
        raise

# Глобальные переменные для моделей
diarization_pipeline = None
processor = None
whisper_model = None
device = None

def initialize_models_fast():
    """Быстрая инициализация моделей"""
    global diarization_pipeline, processor, whisper_model, device
    
    PerformanceOptimizer.optimize_torch()
    
    if diarization_pipeline is None:
        diarization_pipeline = load_pyannote_fast()
        
    if whisper_model is None:
        processor, whisper_model, device = load_whisper_fast()
    
    logging.info("✅ Модели готовы к работе")

# ------------------- Быстрая транскрипция -------------------
def transcribe_fast(audio_path):
    """Быстрая транскрипция"""
    logging.info("🎧 Быстрая транскрипция...")
    start_time = time.time()
    
    try:
        # Используем более быстрые настройки
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            chunk_length_s=20,
            stride_length_s=5,
            generate_kwargs={"language": "russian"},
            device=0 if torch.cuda.is_available() else -1,  # GPU если есть
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        result = asr(audio_path, return_timestamps=True)
        
        PerformanceOptimizer.log_processing_time(start_time, "Транскрипция")
        return result["text"], result["chunks"]

    except Exception as e:
        logging.error(f"❌ Ошибка транскрипции: {e}")
        raise

# ------------------- Оптимизированная обработка -------------------
def process_short_audio_fast(audio_path):
    """Быстрая обработка коротких аудио"""
    logging.info("🔹 Быстрая обработка аудио...")
    start_time = time.time()
    
    if diarization_pipeline is None:
        raise ValueError("Diarization pipeline не инициализирован")

    # Диаризация
    diarization = diarization_pipeline(audio_path)
    
    # Транскрипция
    full_text, chunks = transcribe_fast(audio_path)
    
    # Объединение
    result = align_diarization_and_transcript_fast(diarization, chunks)
    
    PerformanceOptimizer.log_processing_time(start_time, "Обработка аудио")
    return result

def process_long_audio_fast(audio_path, chunk_duration=300):  # Увеличили чанки
    """Быстрая обработка длинных аудио"""
    logging.info("🔸 Быстрая обработка длинного аудио...")
    start_time = time.time()
    
    if diarization_pipeline is None:
        raise ValueError("Diarization pipeline не инициализирован")

    # Загружаем все аудио сразу (быстрее чем по частям)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = len(y) / sr
    logging.info(f"📊 Общая длительность: {total_duration:.1f} сек")

    all_segments = []
    chunk_size = chunk_duration * sr

    for i, start_sample in enumerate(range(0, len(y), chunk_size)):
        chunk_end = min(start_sample + chunk_size, len(y))
        chunk_audio = y[start_sample:chunk_end]

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, chunk_audio, sr)

        try:
            logging.info(f"🔹 Обработка чанка {i+1}...")
            
            # Параллельная обработка не используется для стабильности
            chunk_diarization = diarization_pipeline(temp_path)
            _, chunk_chunks = transcribe_fast(temp_path)
            chunk_result = align_diarization_and_transcript_fast(chunk_diarization, chunk_chunks)

            # Корректируем временные метки
            time_offset = start_sample / sr
            for segment in chunk_result:
                segment["start"] += time_offset
                segment["end"] += time_offset

            all_segments.extend(chunk_result)
            logging.info(f"✅ Чанк {i+1} обработан")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    PerformanceOptimizer.log_processing_time(start_time, "Обработка длинного аудио")
    return all_segments

def align_diarization_and_transcript_fast(diarization, transcript_chunks):
    """Быстрое объединение результатов"""
    segments = []

    for chunk in transcript_chunks:
        start = chunk["timestamp"][0] or 0
        end = chunk["timestamp"][1] or 0
        if start == 0 and end == 0:
            continue

        best_speaker = "unknown"
        max_overlap = 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(start, turn.start)
            overlap_end = min(end, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker

        segments.append({
            "start": float(start),
            "end": float(end),
            "speaker": best_speaker,
            "text": chunk["text"].strip()
        })

    # Быстрое объединение
    merged = []
    for seg in segments:
        if (merged and 
            seg["speaker"] == merged[-1]["speaker"] and 
            seg["start"] <= merged[-1]["end"] + 2.0):  # Увеличили интервал
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    return merged

def send_email(subject, body):
    """Отправка email"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = "test@test.tu"
        msg['To'] = EMAIL_TO
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logging.info(f"📧 Email отправлен: {subject}")
    except Exception as e:
        logging.error(f"Ошибка при отправке email: {e}")

def process_file_fast(s3_key):
    """Быстрая обработка файла"""
    local_path = None
    total_start_time = time.time()
    
    try:
        initialize_models_fast()

        local_path = os.path.join(LOCAL_TMP, os.path.basename(s3_key))
        logging.info(f"⬇️ Загружаем файл S3: {s3_key}")
        s3.download_file(S3_BUCKET, s3_key, local_path)

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logging.info(f"📊 Размер файла: {file_size_mb:.1f} MB")

        # Конвертация
        ext = local_path.rsplit('.', 1)[1].lower()
        if ext != "wav":
            local_path = convert_to_wav_fast(local_path)

        # Определяем длительность
        duration = librosa.get_duration(filename=local_path)
        logging.info(f"⏱ Длительность аудио: {duration:.1f} секунд")

        # Выбираем стратегию
        if duration > 600:  # 10 минут
            result_segments = process_long_audio_fast(local_path)
        else:
            result_segments = process_short_audio_fast(local_path)

        # Результат
        result_json = json.dumps({
            "status": "success",
            "segments": result_segments,
            "total_duration": duration,
            "file": os.path.basename(s3_key),
            "processing_time": round(time.time() - total_start_time, 1)
        }, ensure_ascii=False, indent=2)

        send_email(f"Диаризация {os.path.basename(s3_key)}", result_json)
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
            logging.info(f"🗑 Удален из S3: {s3_key}")
        except Exception as e:
            logging.error(f"Ошибка удаления из S3: {e}")
        
        total_time = time.time() - total_start_time
        logging.info(f"✅ Обработка завершена за {total_time:.1f} сек: {len(result_segments)} сегментов")

    except Exception as e:
        logging.error(f"❌ Ошибка обработки файла {s3_key}: {e}")
        send_email(f"Ошибка диаризации {os.path.basename(s3_key)}", str(e))
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

# ------------------- Фоновый цикл -------------------
async def background_loop():
    """Фоновый цикл"""
    while True:
        try:
            response = s3.list_objects_v2(Bucket=S3_BUCKET)
            objs = response.get("Contents", [])

            if objs:
                logging.info(f"📁 Найдено файлов для обработки: {len(objs)}")

            for obj in objs:
                process_file_fast(obj["Key"])
                await asyncio.sleep(2)  # Уменьшили паузу

        except Exception as e:
            logging.error(f"❌ Ошибка в фоновом цикле: {e}")
            await asyncio.sleep(30)

        await asyncio.sleep(CHECK_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи при старте"""
    logging.info("🚀 Запуск быстрого фонового цикла...")
    asyncio.create_task(background_loop())

# ------------------- FastAPI endpoint -------------------
@app.get("/")
def read_root():
    return {"status": "ok", "optimized_for_speed": True}

@app.get("/health")
def health_check():
    """Проверка здоровья сервиса"""
    device_type = PerformanceOptimizer.get_available_device()
    return {
        "status": "healthy",
        "device": device_type,
        "models_loaded": diarization_pipeline is not None and whisper_model is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/performance")
def performance_info():
    """Информация о производительности"""
    return {
        "device": PerformanceOptimizer.get_available_device(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads()
    }

# ------------------- Запуск -------------------
if __name__ == "__main__":
    asyncio.run(background_loop())