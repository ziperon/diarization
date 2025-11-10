import os
import time
import json
import tempfile
import logging
import subprocess
import smtplib
from email.mime.text import MIMEText
import asyncio
import librosa
import soundfile as sf
from fastapi import FastAPI
import boto3
import torch
from pyannote.audio import Pipeline
import whisper  # ← ДОБАВИТЬ прямой импорт

# ------------------- Конфиги -------------------
S3_BUCKET = "diarization-files"
LOCAL_TMP = os.path.join(os.getcwd(), "tmp", "audiot")
CHECK_INTERVAL = 10
SUPPORTED_EXT = ['mp3', 'm4a', 'wav', 'flac']

EMAIL_HOST = "smtp.mailmug.net"
EMAIL_PORT = 2525
EMAIL_USER = "rv52j9uijrxg83fv"
EMAIL_PASS = "i2qukytuj2hrtunr"
EMAIL_TO = "your-email@gmail.com"

MODELS_DIR = "./models"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v2"  # ← ИСПОЛЬЗУЕМ large-v2 (быстрее чем v3)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------- Логирование -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------- S3 -------------------
s3 = boto3.client(
    "s3",
    endpoint_url='http://127.0.0.1:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
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
            if gpu_memory >= 4:
                logging.info(f"🎯 Используем GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
                return "cuda"
        
        logging.info("🎯 Используем CPU")
        return "cpu"
    
    @staticmethod
    def optimize_torch():
        """Оптимизируем PyTorch для скорости"""
        torch.set_num_threads(min(8, os.cpu_count() or 8))  # ↑ увеличили потоки
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Очищаем кеш CUDA для избежания утечек памяти
            torch.cuda.empty_cache()
    
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
        # Оптимизированные параметры FFmpeg
        cmd = [
            "ffmpeg", "-y", 
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-threads", "4",  # ↑ многопоточность
            "-hide_banner",
            "-loglevel", "error",
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.replace(temp_path, input_path)
        PerformanceOptimizer.log_processing_time(start_time, "Конвертация")
        return input_path
        
    except Exception as e:
        logging.error(f"❌ Ошибка конвертации: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# ------------------- Оптимизированная загрузка моделей -------------------
def load_pyannote_fast():
    """Быстрая загрузка pyannote с оптимизацией"""
    logging.info("⚡ Загружаем pyannote...")
    start_time = time.time()
    
    try:
        hf_token = "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"  # ← НУЖЕН ТОКЕН!
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            use_auth_token=hf_token,
            cache_dir=MODELS_DIR
        )
        
        device_type = PerformanceOptimizer.get_available_device()
        pipeline = pipeline.to(torch.device(device_type))
        
        # Оптимизируем настройки для скорости
        pipeline._segmentation.batch_size = 4  # ↑ батч-сайз
        pipeline._segmentation.device = torch.device(device_type)
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка PyAnnote")
        return pipeline
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки pyannote: {e}")
        raise

def load_whisper_fast():
    """Оптимизированная загрузка Whisper"""
    logging.info("🔄 Загружаем оптимизированный Whisper...")
    start_time = time.time()
    
    try:
        device_type = PerformanceOptimizer.get_available_device()
        
        # Загружаем модель с оптимизацией
        model = whisper.load_model(
            WHISPER_MODEL, 
            device=device_type,
            download_root=MODELS_DIR
        )
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка Whisper")
        return model, device_type
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки Whisper: {e}")
        raise

# Глобальные переменные для моделей
diarization_pipeline = None
whisper_model = None
device = None

def initialize_models_fast():
    """Быстрая инициализация моделей"""
    global diarization_pipeline, whisper_model, device
    
    if diarization_pipeline is None or whisper_model is None:
        preload_all_models()

def preload_all_models():
    """Предварительная загрузка всех моделей"""
    logging.info("🔄 Предзагрузка всех моделей...")
    global diarization_pipeline, whisper_model, device
    
    PerformanceOptimizer.optimize_torch()
    device_type = PerformanceOptimizer.get_available_device()
    
    diarization_pipeline = load_pyannote_fast()
    whisper_model, device = load_whisper_fast()
    
    logging.info("✅ Все модели предзагружены")

# ------------------- ОПТИМИЗИРОВАННАЯ транскрипция -------------------
def transcribe_optimized(audio_path):
    """ОПТИМИЗИРОВАННАЯ транскрипция с лучшими настройками"""
    logging.info("🎧 Оптимизированная транскрипция...")
    start_time = time.time()

    try:
        # ОПТИМАЛЬНЫЕ НАСТРОЙКИ ДЛЯ СКОРОСТИ
        result = whisper_model.transcribe(
            audio_path,
            language="ru",
            fp16=True,  # ВКЛЮЧАЕМ FP16 (2x ускорение на GPU)
            beam_size=3,  # ↓ уменьшаем для скорости
            best_of=2,    # ↓ уменьшаем для скорости  
            temperature=0.0,  # Более стабильные результаты
            no_speech_threshold=0.6,  # Лучше определяет речь
            compression_ratio_threshold=2.4,  # Фильтрация шума
            condition_on_previous_text=False,  # ↑ ускоряет длинные аудио
            word_timestamps=True  # Нужно для диаризации
        )
        
        result_text = ""
        result_chunks = []

        for segment in result["segments"]:
            result_text += segment["text"] + " "
            result_chunks.append({
                "timestamp": [segment["start"], segment["end"]],
                "text": segment["text"].strip()
            })

        PerformanceOptimizer.log_processing_time(start_time, "Транскрипция")
        logging.info(f"📝 Транскрибировано: {len(result_text)} символов, {len(result_chunks)} сегментов")
        return result_text.strip(), result_chunks

    except Exception as e:
        logging.error(f"❌ Ошибка транскрипции: {e}")
        raise

# ------------------- ОПТИМИЗИРОВАННАЯ обработка -------------------
def process_audio_optimized(audio_path):
    """УНИФИЦИРОВАННАЯ оптимизированная обработка"""
    logging.info("🔹 Оптимизированная обработка аудио...")
    start_time = time.time()
    
    if diarization_pipeline is None:
        raise ValueError("Diarization pipeline не инициализирован")

    # Очищаем кеш CUDA перед обработкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Параллельная обработка диаризации и транскрипции
    import threading
    
    diarization_result = [None]
    transcription_result = [None]
    
    def run_diarization():
        diarization_result[0] = diarization_pipeline(audio_path)
    
    def run_transcription():
        transcription_result[0] = transcribe_optimized(audio_path)
    
    # Запускаем в параллельных потоках
    t1 = threading.Thread(target=run_diarization)
    t2 = threading.Thread(target=run_transcription)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    full_text, chunks = transcription_result[0]
    
    # Объединение результатов
    result = align_diarization_and_transcript_fast(diarization_result[0], chunks)
    
    PerformanceOptimizer.log_processing_time(start_time, "Полная обработка")
    return result

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

    # Объединение сегментов
    merged = []
    for seg in segments:
        if (merged and 
            seg["speaker"] == merged[-1]["speaker"] and 
            seg["start"] <= merged[-1]["end"] + 1.5):  # Оптимальный интервал
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    logging.info(f"🎯 Объединено в {len(merged)} сегментов")
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
    """ОПТИМИЗИРОВАННАЯ обработка файла"""
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

        # ВСЕГДА используем оптимизированную обработку
        result_segments = process_audio_optimized(local_path)

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
        speed_ratio = duration / total_time if total_time > 0 else 0
        logging.info(f"✅ Обработка завершена за {total_time:.1f} сек ({speed_ratio:.2f}x реального времени)")

    except Exception as e:
        logging.error(f"❌ Ошибка обработки файла {s3_key}: {e}")
        send_email(f"Ошибка диаризации {os.path.basename(s3_key)}", str(e))
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        # Очищаем память после обработки
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"❌ Ошибка в фоновом цикле: {e}")
            await asyncio.sleep(30)

        await asyncio.sleep(CHECK_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи при старте"""
    logging.info("🚀 Предзагрузка моделей...")
    preload_all_models()
    logging.info("🚀 Запуск оптимизированного фонового цикла...")
    asyncio.create_task(background_loop())

@app.get("/")
def read_root():
    return {"status": "ok", "optimized": True, "version": "2.0"}

@app.get("/health")
def health_check():
    device_type = PerformanceOptimizer.get_available_device()
    return {
        "status": "healthy",
        "device": device_type,
        "models_loaded": diarization_pipeline is not None and whisper_model is not None
    }

if __name__ == "__main__":
    asyncio.run(background_loop())