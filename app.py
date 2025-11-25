import pandas as pd
import os
import time
import json
import tempfile
import logging
import subprocess
import smtplib
from email.mime.text import MIMEText
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional
import librosa
import soundfile as sf
from fastapi import FastAPI
import boto3
import torch
import numpy as np
try:
    setattr(np, "NaN", np.nan)
    setattr(np, "NAN", np.nan)
except Exception:
    pass
from pyannote.audio import Pipeline
import whisper  # ← ДОБАВИТЬ прямой импорт
from gigaam_integration import GigaAMRecognizer
from dion_client import DionApiClient, DionApiError
import settings
import requests
from huggingface_hub import snapshot_download
from datetime import datetime, timedelta
import warnings
from crypto import decrypt_password
from botocore.exceptions import ClientError, EndpointConnectionError

warnings.filterwarnings("ignore")

os.makedirs(settings.LOCAL_TMP, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# ------------------- Логирование -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# ------------------- S3 -------------------
s3 = None
if getattr(settings, "S3_ENABLED", False):
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url='http://localhost:9000',
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            verify=False,
            region_name='us-east-1'
        )
        s3.list_buckets()
        logging.info("✅ Успешно подключились к MinIO!")
    except (ClientError, EndpointConnectionError) as e:
        logging.warning(f"❌ Ошибка подключения к MinIO: {e}. Продолжаем без S3")
        s3 = None
else:
    logging.info("S3/MinIO отключен (S3_ENABLED=0)")

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

        # Используем контекстный менеджер для безопасной загрузки с необходимыми классами
        try:
            from torch.torch_version import TorchVersion
            from pyannote.audio.core.task import Specifications
            with torch.serialization.safe_globals([TorchVersion, Specifications]):
                pipeline = Pipeline.from_pretrained(
                    settings.PYANNOTE_MODEL,
                    cache_dir=settings.MODELS_DIR,
                    local_files_only=True
                )
        except (ImportError, AttributeError, TypeError) as e:
            # Если контекстный менеджер не поддерживается, загружаем обычным способом
            logging.debug(f"Контекстный менеджер не поддерживается, используем обычную загрузку: {e}")
            pipeline = Pipeline.from_pretrained(
                settings.PYANNOTE_MODEL,
                cache_dir=settings.MODELS_DIR,
                use_auth_token="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"
            )
        
        device_type = PerformanceOptimizer.get_available_device()
        pipeline = pipeline.to(torch.device(device_type))
        
        # Оптимизируем настройки для скорости
        pipeline._segmentation.batch_size = 4  # ↑ батч-сайз
        pipeline._segmentation.device = torch.device(device_type)
        
        # Настраиваем параметры для улучшения разделения спикеров
        # Параметры кластеризации - ключевые для разделения разных голосов
        try:
            # В PyAnnote 3.1 параметры кластеризации устанавливаются через атрибуты
            # Проверяем различные возможные пути доступа к параметрам
            if hasattr(pipeline, 'clustering'):
                clustering = pipeline.clustering
                # Устанавливаем threshold (порог кластеризации)
                if hasattr(clustering, 'threshold'):
                    clustering.threshold = settings.DIARIZATION_CLUSTERING_THRESHOLD
                    logging.info(f"🎯 Установлен порог кластеризации: {settings.DIARIZATION_CLUSTERING_THRESHOLD} (меньше = больше спикеров)")
                elif hasattr(clustering, '_threshold'):
                    clustering._threshold = settings.DIARIZATION_CLUSTERING_THRESHOLD
                    logging.info(f"🎯 Установлен порог кластеризации (через _threshold): {settings.DIARIZATION_CLUSTERING_THRESHOLD}")
                
                # Устанавливаем min_cluster_size
                if hasattr(clustering, 'min_cluster_size'):
                    clustering.min_cluster_size = settings.DIARIZATION_MIN_CLUSTER_SIZE
                    logging.info(f"🎯 Установлен минимальный размер кластера: {settings.DIARIZATION_MIN_CLUSTER_SIZE}")
                elif hasattr(clustering, '_min_cluster_size'):
                    clustering._min_cluster_size = settings.DIARIZATION_MIN_CLUSTER_SIZE
                    logging.info(f"🎯 Установлен минимальный размер кластера (через _min_cluster_size): {settings.DIARIZATION_MIN_CLUSTER_SIZE}")
            
            # Параметры сегментации
            if hasattr(pipeline, '_segmentation'):
                seg = pipeline._segmentation
                if hasattr(seg, 'min_duration_off'):
                    seg.min_duration_off = settings.DIARIZATION_MIN_DURATION_OFF
                if hasattr(seg, 'min_duration_on'):
                    seg.min_duration_on = settings.DIARIZATION_MIN_DURATION_ON
            
            # Альтернативный способ: установка через параметры pipeline напрямую
            if hasattr(pipeline, 'instantiate'):
                # Пытаемся обновить параметры через instantiate
                try:
                    # Получаем текущие параметры
                    if hasattr(pipeline, '_params') and pipeline._params:
                        if 'clustering' in pipeline._params:
                            pipeline._params['clustering']['threshold'] = settings.DIARIZATION_CLUSTERING_THRESHOLD
                            pipeline._params['clustering']['min_cluster_size'] = settings.DIARIZATION_MIN_CLUSTER_SIZE
                except Exception as e:
                    logging.debug(f"Не удалось обновить через _params: {e}")
        except Exception as e:
            logging.warning(f"⚠️ Не удалось установить некоторые параметры диаризации: {e}")
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка PyAnnote")
        logging.info(f"✅ PyAnnote настроен для улучшенного разделения спикеров (threshold={settings.DIARIZATION_CLUSTERING_THRESHOLD}, min_cluster_size={settings.DIARIZATION_MIN_CLUSTER_SIZE})")
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
            settings.WHISPER_MODEL, 
            device=device_type,
            download_root=settings.MODELS_DIR
        )
        
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка Whisper")
        return model, device_type
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки Whisper: {e}")
        raise

def load_gigaam_fast():
    """Оптимизированная загрузка GigaAM"""
    logging.info("🔄 Загружаем GigaAM v3...")
    start_time = time.time()
    try:
        device_type = PerformanceOptimizer.get_available_device()
        recognizer = GigaAMRecognizer(model_type=settings.GIGAAM_MODEL_TYPE, device=device_type)
        recognizer.load_model()
        PerformanceOptimizer.log_processing_time(start_time, "Загрузка GigaAM")
        return recognizer, device_type
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки GigaAM: {e}")
        # raise

# Глобальные переменные для моделей
diarization_pipeline = None
gigaam_recognizer = None
whisper_model = None
device = None
# Lock для синхронизации доступа к моделям при параллельной обработке
models_lock = threading.Lock()

def initialize_models_fast():
    """Быстрая инициализация моделей"""
    global diarization_pipeline, gigaam_recognizer, whisper_model, device
    
    if diarization_pipeline is None or (gigaam_recognizer is None and whisper_model is None):
        preload_all_models()

def preload_all_models():
    """Предварительная загрузка всех моделей"""
    logging.info("🔄 Предзагрузка всех моделей...")
    global diarization_pipeline, gigaam_recognizer, whisper_model, device

    patch_torch_for_weights_only()

    PerformanceOptimizer.optimize_torch()
    device_type = PerformanceOptimizer.get_available_device()
    
    diarization_pipeline = load_pyannote_fast()
    # Primary: GigaAM
    gigaam_recognizer, device = load_gigaam_fast()
    # Fallback: Whisper (не критично, если не загрузится)
    # try:
    #     whisper_model, _ = load_whisper_fast()
    # except Exception as e:
    #     logging.warning(f"⚠️ Whisper fallback недоступен: {e}")
    
    logging.info("✅ Все модели предзагружены")

def _format_segments_from_gigaam(result: dict):
    result_text = " ".join(seg.get("text", "") for seg in result.get("segments", [])) or result.get("text", "")
    result_chunks = []
    for seg in result.get("segments", []):
        result_chunks.append({
            "timestamp": [seg.get("start", 0), seg.get("end", 0)],
            "text": seg.get("text", "").strip()
        })
    return result_text.strip(), result_chunks

def _transcribe_with_whisper(audio_path: str):
    if settings.TRANSCRIPTION_MODE == "quality":
        result = whisper_model.transcribe(
            audio_path,
            language="ru",
            fp16=True,
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            temperature=0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            logprob_threshold=-1.0,
            initial_prompt="Это запись разговора на русском языке. "
        )
    else:
        result = whisper_model.transcribe(
            audio_path,
            language="ru",
            fp16=True,
            word_timsestamps=True,
            beam_size=3,
            best_of=2,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
        )
    result_text = ""
    result_chunks = []
    for segment in result["segments"]:
        result_text += segment["text"] + " "
        result_chunks.append({
            "timestamp": [segment["start"], segment["end"]],
            "text": segment["text"].strip()
        })
    return result_text.strip(), result_chunks

def _transcribe_with_gigaam(audio_path: str):
    result = gigaam_recognizer.transcribe(audio_path, language="ru")
    return _format_segments_from_gigaam(result)

# ------------------- ОПТИМИЗИРОВАННАЯ транскрипция -------------------
def transcribe_optimized(audio_path):
    """ОПТИМИЗИРОВАННАЯ транскрипция с приоритетом GigaAM и fallback на Whisper"""
    logging.info(f"🎧 Транскрипция (primary: {settings.ASR_PRIMARY})...")
    start_time = time.time()

    def try_gigaam():
        if gigaam_recognizer is None:
            raise RuntimeError("GigaAM не инициализирован")
        return _transcribe_with_gigaam(audio_path)

    def try_whisper():
        if whisper_model is None:
            raise RuntimeError("Whisper не инициализирован")
        return _transcribe_with_whisper(audio_path)

    try:
        if settings.ASR_PRIMARY == "gigaam":
            full_text, chunks = try_gigaam()
            logging.info(f"✅ GigaAM транскрипция завершена за {time.time() - start_time:.1f} сек")
            logging.info(f"Text: {full_text} Chunk: {chunks}")
        else:
            full_text, chunks = try_whisper()
    except Exception as primary_err:
        logging.warning(f"⚠️ Primary ASR failed: {primary_err}. Пытаемся fallback...")
        if settings.ASR_PRIMARY == "gigaam":
            full_text, chunks = try_whisper()
        else:
            full_text, chunks = try_gigaam()

    PerformanceOptimizer.log_processing_time(start_time, "Транскрипция")
    logging.info(f"📝 Транскрибировано: {len(full_text)} символов, {len(chunks)} сегментов")
    return full_text, chunks

# ------------------- ОПТИМИЗИРОВАННАЯ обработка -------------------
def process_audio_optimized(audio_path, tracks=None, users_info=None):
    """УНИФИЦИРОВАННАЯ оптимизированная обработка"""
    logging.info("🔹 Оптимизированная обработка аудио...")
    start_time = time.time()
    
    # Проверяем инициализацию моделей (PyTorch модели thread-safe для inference)
    if diarization_pipeline is None or (gigaam_recognizer is None and whisper_model is None):
        raise ValueError("Модели не инициализированы")

    # Очищаем кеш CUDA перед обработкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Параллельная обработка диаризации и транскрипции
    # PyTorch модели поддерживают параллельные inference запросы без дополнительной синхронизации
    diarization_result = [None]
    transcription_result = [None]
    
    def run_diarization():
        # Вызываем диаризацию с настройками для улучшенного разделения спикеров
        # Параметры можно передать через словарь
        diarization_params = {
            "clustering": {
                "threshold": settings.DIARIZATION_CLUSTERING_THRESHOLD,
                "min_cluster_size": settings.DIARIZATION_MIN_CLUSTER_SIZE
            },
            "segmentation": {
                "min_duration_off": settings.DIARIZATION_MIN_DURATION_OFF,
                "min_duration_on": settings.DIARIZATION_MIN_DURATION_ON
            }
        }
        try:
            # Пытаемся передать параметры через kwargs
            diarization_result[0] = diarization_pipeline(audio_path, **diarization_params)
        except TypeError:
            # Если не поддерживается, используем обычный вызов
            # Параметры уже установлены при загрузке pipeline
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
     # Используем УЛУЧШЕННОЕ объединение с контекстом
    result = align_diarization_and_transcript_contextual(
        diarization_result, chunks, tracks, users_info
    )
    
    PerformanceOptimizer.log_processing_time(start_time, "Полная обработка")
    return result

def parse_tracks_json(json_path):
    """Парсинг JSON файла с треками (каждая строка - отдельный JSON объект)"""
    tracks = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    track = json.loads(line)
                    # Конвертируем миллисекунды в секунды
                    track['start_s'] = track['start_ms'] / 1000.0
                    track['end_s'] = track['end_ms'] / 1000.0
                    tracks.append(track)
                except json.JSONDecodeError as e:
                    logging.warning(f"⚠️ Ошибка парсинга строки JSON: {e}")
                    continue
        logging.info(f"📋 Загружено {len(tracks)} треков из JSON")
        return tracks
    except Exception as e:
        logging.error(f"❌ Ошибка чтения JSON файла: {e}")
        return []

def get_user_id_for_time_advanced(tracks, start_time, end_time, previous_segments=None, speaker_history=None):
    """Продвинутый алгоритм с учетом контекста и истории спикеров"""
    
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    segment_duration_ms = (end_time - start_time) * 1000
    
    # Кандидаты и их оценки
    candidates = {}
    
    for track in tracks:
        track_start = track['start_ms']
        track_end = track['end_ms']
        user_id = track['user_id']
        
        # Базовое перекрытие
        overlap_start = max(start_ms, track_start)
        overlap_end = min(end_ms, track_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration == 0:
            continue
            
        # Процент перекрытия относительно сегмента и трека
        overlap_pct_segment = overlap_duration / segment_duration_ms if segment_duration_ms > 0 else 0
        overlap_pct_track = overlap_duration / (track_end - track_start) if (track_end - track_start) > 0 else 0
        
        # Взвешенная оценка перекрытия
        overlap_score = (overlap_pct_segment * 0.6 + overlap_pct_track * 0.4)
        
        # Штраф за временное несоответствие
        time_penalty = 0
        if overlap_pct_segment < 0.3:  # Малое перекрытие
            time_penalty = 0.3
        elif abs(track_start - start_ms) > 2000:  # Большой разрыв по началу
            time_penalty = 0.2
            
        final_score = max(0, overlap_score - time_penalty)
        
        if user_id not in candidates or final_score > candidates[user_id]['score']:
            candidates[user_id] = {
                'score': final_score,
                'overlap_pct': overlap_pct_segment,
                'track': track
            }
    
    # Применяем контекстные правила
    best_candidate = None
    best_score = 0
    
    for user_id, data in candidates.items():
        score = data['score']
        
        # Увеличиваем оценку если этот user_id часто встречался недавно
        if speaker_history and user_id in speaker_history:
            recent_frequency = speaker_history.get(user_id, 0)
            score += min(0.2, recent_frequency * 0.1)  # Бонус до 20%
        
        # Увеличиваем оценку для коротких сегментов, если они близки к предыдущему
        if (segment_duration_ms < 3000 and previous_segments and 
            len(previous_segments) > 0):
            last_segment = previous_segments[-1]
            if (last_segment.get('user_id') == user_id and 
                start_time - last_segment['end'] < 3.0):
                score += 0.15  # Бонус за последовательные короткие реплики
        
        if score > best_score:
            best_score = score
            best_candidate = user_id
    
    # Порог принятия решения
    if best_candidate and best_score > 0.2:
        logging.debug(f"✅ user_id {best_candidate} для {start_time:.2f}-{end_time:.2f} (score: {best_score:.2f})")
        return best_candidate
    
    # Если нет хороших кандидатов, используем эвристики
    return get_user_id_contextual(tracks, start_time, end_time, previous_segments)

def parse_event_path_and_get_range(path: str) -> tuple[str, str, str]:
    """
    Пример входа:
        '/f6c28cf1-4c6c-44b4-a670-35158d9798a0/2025-11-10T11-21-00'

    Возвращает:
        event_id, time_start, time_end
        (в ISO8601 формате: yyyy-mm-ddTHH:MM:SSZ)
    """
    # Убираем лишние пробелы и слеши по краям
    clean = path.strip().strip("/")

    parts = clean.split("/")
    if len(parts) != 2:
        raise ValueError(f"Некорректный путь: {path}")

    event_id = parts[0]
    raw_dt = parts[1]

    # Пример: 2025-11-10T11-21-00 → приводим к datetime
    dt = datetime.strptime(raw_dt, "%Y-%m-%dT%H-%M-%S")

    # Вычисляем диапазон ±5 часов
    time_start = dt - timedelta(hours=2)
    time_end = dt + timedelta(hours=2)

    # Возвращаем ISO8601 в UTC
    return (
        event_id,
        time_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        time_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

def get_user_id_contextual(tracks, start_time, end_time, previous_segments=None):
    """Контекстные эвристики с учетом предыдущих сегментов"""
    
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    
    # 1. Предыдущий спикер (для коротких реплик)
    if previous_segments and len(previous_segments) > 0:
        last_segment = previous_segments[-1]
        last_user_id = last_segment.get('user_id')
        last_end = last_segment['end']
        
        # Если короткая реплика сразу после предыдущей
        if (end_time - start_time < 2.0 and  # Короткая реплика
            start_time - last_end < 2.0 and   # Сразу после предыдущей
            last_user_id):
            
            # Проверяем, есть ли треки этого пользователя вблизи
            for track in tracks:
                if (track['user_id'] == last_user_id and
                    abs(track['start_ms'] - start_ms) < 3000):
                    logging.debug(f"🎯 Контекст: короткая реплика от предыдущего спикера {last_user_id}")
                    return last_user_id
    
    # 2. Ближайший трек по времени с учетом типа сегмента
    closest_track = None
    min_diff = float('inf')
    
    for track in tracks:
        # Разница по началу
        start_diff = abs(track['start_ms'] - start_ms)
        # Разница по середине
        mid_track = (track['start_ms'] + track['end_ms']) / 2
        mid_segment = (start_ms + end_ms) / 2
        mid_diff = abs(mid_track - mid_segment)
        
        # Используем минимальную разницу
        diff = min(start_diff, mid_diff)
        
        if diff < min_diff:
            min_diff = diff
            closest_track = track
    
    # 3. Принимаем решение на основе близости
    if closest_track and min_diff < 3000:  # 3 секунды
        user_id = closest_track['user_id']
        
        # Дополнительная проверка: смотрим на длительность трека
        track_duration = closest_track['end_ms'] - closest_track['start_ms']
        segment_duration = end_ms - start_ms
        
        # Если трек достаточно длинный для этого сегмента
        if track_duration >= segment_duration * 0.5:
            logging.debug(f"🎯 Контекст: ближайший трек {user_id} (diff: {min_diff/1000:.2f}s)")
            return user_id
    
    # 4. Статистика по пользователям в временном окне
    user_durations = {}
    time_window_start = start_ms - 5000  # 5 секунд до
    time_window_end = end_ms + 5000      # 5 секунд после
    
    for track in tracks:
        if (track['end_ms'] > time_window_start and 
            track['start_ms'] < time_window_end):
            
            user_id = track['user_id']
            # Вычисляем перекрытие с временным окном
            overlap_start = max(track['start_ms'], time_window_start)
            overlap_end = min(track['end_ms'], time_window_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            user_durations[user_id] = user_durations.get(user_id, 0) + overlap_duration
    
    if user_durations:
        # Выбираем пользователя с наибольшим суммарным временем в окне
        best_user = max(user_durations.items(), key=lambda x: x[1])[0]
        total_duration = sum(user_durations.values())
        confidence = user_durations[best_user] / total_duration if total_duration > 0 else 0
        
        if confidence > 0.4:  # Хотя бы 40% времени в окне
            logging.debug(f"🎯 Контекст: доминирующий пользователь {best_user} (confidence: {confidence:.2f})")
            return best_user
    
    logging.debug(f"⚠️ Не удалось определить user_id для {start_time:.2f}-{end_time:.2f}")
    return None

def create_speaker_to_user_mapping(diarization_annotation, tracks, transcript_chunks):
    """Создает mapping между спикерами диаризации и user_id из tracks"""
    
    speaker_user_mapping = {}
    speaker_scores = {}
    
    # Проходим по всем сегментам диаризации
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        speaker_start = turn.start
        speaker_end = turn.end
        
        # Ищем подходящий user_id из tracks для этого сегмента диаризации
        best_user_id = get_user_id_for_time_advanced(tracks, speaker_start, speaker_end)
        
        if best_user_id:
            if speaker not in speaker_scores:
                speaker_scores[speaker] = {}
            
            if best_user_id not in speaker_scores[speaker]:
                speaker_scores[speaker][best_user_id] = 0
            
            # Увеличиваем счетчик для этой пары speaker-user_id
            duration = speaker_end - speaker_start
            speaker_scores[speaker][best_user_id] += duration
    
    # Для каждого спикера выбираем user_id с наибольшим накопленным временем
    for speaker, user_scores in speaker_scores.items():
        if user_scores:
            best_user_id = max(user_scores.items(), key=lambda x: x[1])[0]
            total_time = sum(user_scores.values())
            confidence = user_scores[best_user_id] / total_time if total_time > 0 else 0
            
            if confidence >= 0.3:  # Минимальная уверенность 30%
                speaker_user_mapping[speaker] = best_user_id
                logging.info(f"🔗 Спикер {speaker} → user_id {best_user_id} (уверенность: {confidence:.2f})")
            else:
                logging.warning(f"⚠️ Низкая уверенность для спикера {speaker}: {confidence:.2f}")
    
    return speaker_user_mapping

def align_diarization_and_transcript_contextual(diarization, transcript_chunks, tracks=None, users_info=None):
    """Объединение результатов с учетом контекста"""
    segments = []
    speaker_history = {}
    previous_segments = []
    
    logging.info(f"🔹 Diarization type: {type(diarization)}")
    
    # ИЗВЛЕКАЕМ объект Annotation из списка
    if isinstance(diarization, list) and len(diarization) > 0:
        logging.info("🎯 Извлекаем объект Annotation из списка")
        diarization_annotation = diarization[0]
    else:
        diarization_annotation = diarization
    
    logging.info(f"🔹 Diarization annotation type: {type(diarization_annotation)}")
    
    # СОЗДАЕМ MAPPING между спикерами и user_id с БАЛАНСИРОВАННЫМ подходом
    speaker_user_mapping = {}
    if tracks:
        speaker_user_mapping = create_speaker_to_user_mapping_balanced(
            diarization_annotation, tracks, transcript_chunks
        )
        logging.info(f"🔗 Создано {len(speaker_user_mapping)} mappings спикер→user_id")
    
    for chunk in transcript_chunks:
        start = chunk["timestamp"][0] or 0
        end = chunk["timestamp"][1] or 0
        
        if start == 0 and end == 0:
            continue
        
        # Определяем спикера из диаризации
        best_speaker = "unknown"
        max_overlap = 0
        
        try:
            for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker
                    
        except Exception as e:
            logging.error(f"❌ Ошибка при определении спикера: {e}")
            best_speaker = "SPEAKER_00"
        
        # ОПРЕДЕЛЯЕМ user_id через mapping
        user_id = None
        if tracks:
            if best_speaker in speaker_user_mapping:
                user_id = speaker_user_mapping[best_speaker]
                method = "speaker_mapping"
            else:
                # Резервный метод для неизвестных спикеров
                user_id = get_user_id_for_time_advanced(
                    tracks, start, end, previous_segments, speaker_history
                )
                method = "time_overlap"
            
            if user_id:
                logging.debug(f"🎯 user_id {user_id} для сегмента {start:.2f}-{end:.2f} (метод: {method})")
                speaker_history[user_id] = speaker_history.get(user_id, 0) + 1

        segment = {
            "start": float(start),
            "end": float(end),
            "speaker": best_speaker,
            "text": chunk["text"].strip()
        }
        
        if user_id:
            #segment["user_id"] = user_id
            if users_info and user_id in users_info and users_info[user_id]:
                #segment["user_info"] = users_info[user_id]
                segment["speaker_name"] = users_info[user_id]["name"]
        
        segments.append(segment)
        previous_segments.append(segment)
        
        if len(previous_segments) > 10:
            previous_segments.pop(0)

    # Объединение сегментов
    merged = []
    for seg in segments:
        if (merged and 
            seg["speaker"] == merged[-1]["speaker"] and
            seg.get("user_id") == merged[-1].get("user_id") and
            seg["start"] <= merged[-1]["end"] + 2.0):
            
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    # Финальная статистика
    if tracks:
        user_stats = {}
        speaker_stats = {}
        
        for seg in merged:
            if 'user_id' in seg:
                user_stats[seg['user_id']] = user_stats.get(seg['user_id'], 0) + 1
            speaker_stats[seg['speaker']] = speaker_stats.get(seg['speaker'], 0) + 1
        
        logging.info(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        logging.info(f"   User_id распределение: {user_stats}")
        logging.info(f"   Спикеры распределение: {speaker_stats}")
        logging.info(f"   Сегментов с user_id: {len([s for s in merged if 'user_id' in s])}/{len(merged)}")

    return merged

def get_users_info(user_ids, dion_client=None, event_id=None, time_start=None, time_end=None):
    """
    Получить информацию о пользователях из DION API с кешированием, используя get_event_users.

    Аргументы:
        user_ids: список user_id, которых нужно выбрать из ответа
        dion_client: экземпляр DionApiClient
        event_id: UUID события для запроса
        time_start: начало периода ISO8601
        time_end: конец периода ISO8601
    Возвращает:
        dict: {user_id: user_info или None}
    """

    if not dion_client or not user_ids or not event_id:
        return {}

    unique_user_ids = set(user_ids)
    users_info = {}

    logging.info(f"👤 Запрашиваем пользователей события {event_id} из DION API")

    try:
        # 1. Запрос всех пользователей события за период
        response = dion_client.get_event_users(
            event_id=event_id,
            time_start=time_start,
            time_end=time_end
        )

        event_users = response.get("users", [])
        logging.info(f"📁 Получено {len(event_users)} пользователей из DION API")

        # 2. Сопоставляем только нужные user_id из tracks
        for u in event_users:
            uid = u.get("user_id")
            if uid in unique_user_ids:
                users_info[uid] = {
                    "name": u.get("name"),
                    "email": u.get("email"),
                    "position": u.get("position"),
                    "sessions": u.get("sessions", [])
                }

        # 3. Для отсутствующих пользователей ставим None
        for uid in unique_user_ids:
            if uid not in users_info:
                logging.warning(f"⚠️ В DION нет данных по user_id={uid}")
                users_info[uid] = None

    except DionApiError as e:
        logging.error(f"❌ Ошибка DION API: {e}")
        for uid in unique_user_ids:
            users_info[uid] = None
    except Exception as e:
        logging.error(f"❌ Общая ошибка при запросе пользователей: {e}")
        for uid in unique_user_ids:
            users_info[uid] = None

    return users_info



def align_diarization_and_transcript_fast(diarization, transcript_chunks, tracks=None, users_info=None):
    """Быстрое объединение результатов с добавлением user_id из JSON и информации о пользователях"""
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

        # Получаем user_id из JSON треков, если они есть
        user_id = None
        if tracks:
            user_id = settings.get_user_id_for_time(tracks, start, end)

        segment = {
            "start": float(start),
            "end": float(end),
            "speaker": best_speaker,
            "text": chunk["text"].strip()
        }
        
        if user_id:
            segment["user_id"] = user_id
            # Добавляем информацию о пользователе, если доступна
            if users_info and user_id in users_info and users_info[user_id]:
                segment["user_info"] = users_info[user_id]
        
        segments.append(segment)

    # Объединение сегментов
    merged = []
    for seg in segments:
        if (merged and 
            seg["speaker"] == merged[-1]["speaker"] and
            seg.get("user_id") == merged[-1].get("user_id") and
            seg["start"] <= merged[-1]["end"] + 1.5):  # Оптимальный интервал
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    logging.info(f"🎯 Объединено в {len(merged)} сегментов")
    return merged

def format_segments_to_lines(segments):
    """
    Преобразует массив сегментов (список словарей) в построчный текст.
    
    :param segments: list[dict] — список сегментов с полями:
                     - start (float)
                     - end (float)
                     - speaker (str, опционально)
                     - text (str)
    :return: str — готовая строка в формате:
               [00:00,72 — 00:10,98] Имя: Текст
    """
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02}:{secs:05.2f}".replace('.', ',')

    lines = []
    for seg in segments:
        speaker = seg.get('speaker_name', '').strip() or "Неизвестный"
        text = seg.get('text', '').strip()

        line = f"{speaker}: {text}"
        lines.append(line)

    return '\n'.join(lines)

def send_email(subject: str, body: str, to_email: str = None):
    """
    Отправка email
    
    Args:
        subject: Тема письма
        body: Тело письма (может быть HTML)
        to_email: Email получателя (если None, берется из настроек)
    """
    recipient_email = to_email or settings.EMAIL_TO
    
    msg = MIMEText(body, 'html' if '<' in body else 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = settings.EMAIL_FROM
    msg['To'] = recipient_email
    
    with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
        if settings.EMAIL_USE_TLS:
            server.starttls()
        server.login(settings.EMAIL_USER, decrypt_password(settings.EMAIL_PASS))
        server.send_message(msg)
    
    logging.info(f"📧 Email отправлен: {subject} -> {recipient_email}")
        
    

def create_speaker_to_user_mapping_balanced(diarization_annotation, tracks, transcript_chunks):
    """Балансирует уникальность user_id и уверенность сопоставления"""
    
    speaker_user_mapping = {}
    speaker_scores = {}
    
    logging.info("🎯 Начинаем создание mapping между спикерами и user_id...")
    
    # Сбор статистики по всем сегментам диаризации
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        speaker_start = turn.start
        speaker_end = turn.end
        
        # Ищем лучший user_id для этого сегмента диаризации
        best_user_id = get_user_id_for_time_advanced(tracks, speaker_start, speaker_end)
        
        if best_user_id:
            if speaker not in speaker_scores:
                speaker_scores[speaker] = {}
            
            # Суммируем время для каждой пары спикер-user_id
            duration = speaker_end - speaker_start
            speaker_scores[speaker][best_user_id] = speaker_scores[speaker].get(best_user_id, 0) + duration
    
    # Логируем собранную статистику
    for speaker, user_scores in speaker_scores.items():
        total_time = sum(user_scores.values())
        logging.info(f"📊 Спикер {speaker}: {len(user_scores)} кандидатов, общее время {total_time:.1f}с")
        for user_id, time in user_scores.items():
            confidence = time / total_time
            logging.info(f"   👤 {user_id}: {time:.1f}с ({confidence:.1%})")
    
    # Подготовка кандидатов для каждого спикера
    speaker_candidates = {}
    for speaker, user_scores in speaker_scores.items():
        total_time = sum(user_scores.values())
        candidates = []
        for user_id, time in user_scores.items():
            confidence = time / total_time if total_time > 0 else 0
            candidates.append((user_id, confidence, time))
        
        # Сортируем по уверенности (от высокой к низкой)
        candidates.sort(key=lambda x: x[1], reverse=True)
        speaker_candidates[speaker] = candidates
        
        if candidates:
            best_user, best_conf, best_time = candidates[0]
            logging.info(f"🎯 Спикер {speaker}: лучший кандидат {best_user} ({best_conf:.1%})")
    
    # Многораундовое назначение с приоритетом уникальности
    
    assigned_users = set()  # Уже назначенные user_id
    
    # Раунд 1: назначаем уникальные пары с ВЫСОКОЙ уверенностью (> 0.7)
    logging.info("🔹 Раунд 1: Назначение с высокой уверенностью (> 70%)")
    for speaker, candidates in speaker_candidates.items():
        if speaker in speaker_user_mapping:
            continue
            
        for user_id, confidence, time in candidates:
            if user_id not in assigned_users and confidence > 0.7:
                speaker_user_mapping[speaker] = user_id
                assigned_users.add(user_id)
                logging.info(f"✅ [Раунд 1] {speaker} → {user_id} (уверенность: {confidence:.1%})")
                break
    
    # Раунд 2: назначаем уникальные пары с СРЕДНЕЙ уверенностью (> 0.5)
    logging.info("🔹 Раунд 2: Назначение со средней уверенностью (> 50%)")
    for speaker, candidates in speaker_candidates.items():
        if speaker in speaker_user_mapping:
            continue
            
        for user_id, confidence, time in candidates:
            if user_id not in assigned_users and confidence > 0.5:
                speaker_user_mapping[speaker] = user_id
                assigned_users.add(user_id)
                logging.info(f"✅ [Раунд 2] {speaker} → {user_id} (уверенность: {confidence:.1%})")
                break
    
    # Раунд 3: назначаем уникальные пары с МИНИМАЛЬНОЙ уверенностью (> 0.3)
    logging.info("🔹 Раунд 3: Назначение с минимальной уверенностью (> 30%)")
    for speaker, candidates in speaker_candidates.items():
        if speaker in speaker_user_mapping:
            continue
            
        for user_id, confidence, time in candidates:
            if user_id not in assigned_users and confidence > 0.3:
                speaker_user_mapping[speaker] = user_id
                assigned_users.add(user_id)
                logging.info(f"✅ [Раунд 3] {speaker} → {user_id} (уверенность: {confidence:.1%})")
                break
    
    # Раунд 4: УЛУЧШЕННЫЙ - ищем любого свободного кандидата, даже с низкой уверенностью
    logging.info("🔹 Раунд 4: Поиск любого свободного кандидата")
    for speaker, candidates in speaker_candidates.items():
        if speaker in speaker_user_mapping:
            continue
            
        # Ищем первого свободного кандидата (независимо от уверенности)
        assigned = False
        for user_id, confidence, time in candidates:
            if user_id not in assigned_users:
                speaker_user_mapping[speaker] = user_id
                assigned_users.add(user_id)
                logging.info(f"✅ [Раунд 4] {speaker} → {user_id} (свободный, уверенность: {confidence:.1%})")
                assigned = True
                break
        
        # Если не нашли свободного, берем лучшего доступного с минимальным дублированием
        if not assigned and candidates:
            # Ищем кандидата с наименьшим количеством текущих назначений
            user_assignment_count = {}
            for user_id, confidence, time in candidates:
                # Считаем сколько раз этот user_id уже назначен
                count = sum(1 for s, uid in speaker_user_mapping.items() if uid == user_id)
                user_assignment_count[user_id] = count
            
            # Берем кандидата с минимальным количеством назначений
            best_user_id = min(user_assignment_count.items(), key=lambda x: x[1])[0]
            confidence = next((conf for uid, conf, t in candidates if uid == best_user_id), 0)
            current_count = user_assignment_count[best_user_id]
            
            speaker_user_mapping[speaker] = best_user_id
            
            if current_count > 0:
                logging.warning(f"⚠️ [Раунд 4] {speaker} → {best_user_id} (дублирование #{current_count + 1}, уверенность: {confidence:.1%})")
            else:
                assigned_users.add(best_user_id)
                logging.info(f"✅ [Раунд 4] {speaker} → {best_user_id} (уверенность: {confidence:.1%})")
    
    # Финальная статистика
    logging.info("📈 ФИНАЛЬНАЯ СТАТИСТИКА MAPPING:")
    user_speaker_count = {}
    for speaker, user_id in speaker_user_mapping.items():
        user_speaker_count[user_id] = user_speaker_count.get(user_id, 0) + 1
        confidence = next((conf for uid, conf, t in speaker_candidates[speaker] if uid == user_id), 0)
        logging.info(f"   {speaker} → {user_id} (уверенность: {confidence:.1%})")
    
    # Логируем дублирования
    duplicates = {user: count for user, count in user_speaker_count.items() if count > 1}
    if duplicates:
        logging.warning(f"⚠️ Обнаружены дублирования user_id: {duplicates}")
    else:
        logging.info("✅ Все user_id назначены уникально!")
    
    logging.info(f"🎯 Итог: {len(speaker_user_mapping)} спикеров сопоставлено с {len(assigned_users)} user_id")
    
    return speaker_user_mapping

def process_directory(s3_prefix):
    """Обработка директории с UUID/timestamp структурой"""
    local_audio_path = None
    local_json_path = None
    total_start_time = time.time()
    
    try:
        # Убеждаемся, что модели загружены (с синхронизацией)
        with models_lock:
            initialize_models_fast()
        
        logging.info(f"📁 Обрабатываем директорию: {s3_prefix}")
        
        # Получаем список файлов в директории
        response = s3.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=s3_prefix)
        files = response.get("Contents", [])
        
        # Ищем mp4 и json файлы
        mp4_file = None
        json_file = None
        
        for file_obj in files:
            key = file_obj["Key"]
            filename = os.path.basename(key)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            
            if ext == 'mp4':
                mp4_file = key
            elif ext == 'json':
                json_file = key
        
        if not mp4_file:
            logging.warning(f"⚠️ MP4 файл не найден в {s3_prefix}")
            return
        
        # Загружаем MP4 файл
        local_audio_path = os.path.join(settings.LOCAL_TMP, str(hash(mp4_file)) + ".mp4")
        logging.info(f"⬇️ Загружаем MP4 файл: {mp4_file}")
        s3.download_file(settings.S3_BUCKET, mp4_file, local_audio_path)
        
        file_size_mb = os.path.getsize(local_audio_path) / (1024 * 1024)
        logging.info(f"📊 Размер файла: {file_size_mb:.1f} MB")
        
        # Загружаем JSON файл, если есть
        tracks = None
        if json_file:
            local_json_path = os.path.join(settings.LOCAL_TMP, str(hash(json_file)) + ".json")
            logging.info(f"⬇️ Загружаем JSON файл: {json_file}")
            s3.download_file(settings.S3_BUCKET, json_file, local_json_path)
            tracks = parse_tracks_json(local_json_path)
        else:
            logging.info("ℹ️ JSON файл не найден, работаем без user_id")
        
        # Конвертация
        ext = local_audio_path.rsplit('.', 1)[1].lower()
        if ext != "wav":
            local_audio_path = convert_to_wav_fast(local_audio_path)
        
        # Определяем длительность
        duration = librosa.get_duration(filename=local_audio_path)
        logging.info(f"⏱ Длительность аудио: {duration:.1f} секунд")
        
        # Получаем информацию о пользователях из DION API, если включено
        users_info = {}
        dion_client = None
        owner_email = None
        slug = None

        # Берём диапазон по дате события (например ±5 часов)
        event_id, time_start, time_end = parse_event_path_and_get_range(s3_prefix)
        if settings.DION_API_ENABLED and tracks:
            try:
                dion_client = DionApiClient(access_token=decrypt_password(settings.DION_ACCESS_TOKEN))
                # Собираем все уникальные user_id из треков
                user_ids = [track['user_id'] for track in tracks if 'user_id' in track]


                if user_ids:
                    users_info = get_users_info(
                        user_ids=user_ids,
                        dion_client=dion_client,
                        event_id=event_id,
                        time_start=time_start,
                        time_end=time_end
                    )

                 # ИЗВЛЕКАЕМ UUID события из s3_prefix
                # s3_prefix имеет формат: uuid/timestamp/
                event_uuid = extract_event_uuid_from_s3_prefix(s3_prefix)
                
                if event_uuid:
                    # Получаем информацию о событии
                    event_data = dion_client.get_event_data_by_id(event_uuid)
                    slug = event_data.get("link_settings", {}).get("slug","")
                    logging.info(f"📋 Получены данные события: {event_uuid}")
                    
                    # Извлекаем owner_email из ответа
                    if event_data:
                       owner_email = event_data.get('owner_email')
                       logging.info(f"👤 Owner email: {owner_email}")
            except Exception as e:
                logging.error(f"❌ Ошибка при инициализации DION API клиента: {e}")
                users_info = {}
        
        # ВСЕГДА используем оптимизированную обработку с треками и информацией о пользователях
        result_segments = process_audio_optimized(local_audio_path, tracks, users_info)
        
        # Закрываем DION API клиент
        if dion_client:
            try:
                dion_client.close()
            except Exception:
                pass
        
        # Результат
        result_json = json.dumps({
            "status": "success",
            "segments": result_segments,
            "total_duration": duration,
            "directory": s3_prefix,
            "processing_time": round(time.time() - total_start_time, 1)
        }, ensure_ascii=False, indent=2)
        #logging.info(f"Результат: {result_json}")
        
          # ОТПРАВКА РЕЗУЛЬТАТА НА ПОЧТУ
        if owner_email:
            logging.info(f"Отправка результата на почту {format_segments_to_lines(result_segments)}")
            send_email(f"расшифровка dion-конференции за {iso8601_to_dd_mm_yyyy(time_start)} комната {slug!r}", format_segments_to_lines(result_segments), to_email=owner_email)
        else:
            raise Exception(f"Не найдена почта владельная по {s3_prefix}")
           
        
        # Удаляем только timestamp директорию из S3 (не всю UUID директорию)
        # s3_prefix имеет формат: uuid/timestamp/
        try:
            # Убеждаемся, что префикс заканживается на '/'
            delete_prefix = s3_prefix if s3_prefix.endswith('/') else s3_prefix + '/'
            
            # Получаем список всех объектов с данным префиксом
            objects_to_delete = []
            paginator = s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=delete_prefix):
                if 'Contents' in page:
                    objects_to_delete.extend([{'Key': obj['Key']} for obj in page['Contents']])
            
            # Удаляем все объекты пачкой
            if objects_to_delete:
                response = s3.delete_objects(
                    Bucket=settings.S3_BUCKET,
                    Delete={'Objects': objects_to_delete}
                )
                deleted_count = len(response.get('Deleted', []))
            else:
                deleted_count = 0
            
            logging.info(f"🗑 Удалена timestamp директория {s3_prefix} из S3 (удалено объектов: {deleted_count})")
            
        except Exception as e:
            logging.error(f"Ошибка удаления директории из S3: {e}")
        
        total_time = time.time() - total_start_time
        speed_ratio = duration / total_time if total_time > 0 else 0
        logging.info(f"✅ Обработка завершена за {total_time:.1f} сек ({speed_ratio:.2f}x реального времени)")
    
    except Exception as e:
        logging.error(f"❌ Ошибка обработки директории {s3_prefix}: {e}")
        send_email(f"Ошибка диаризации {s3_prefix}", str(e))
    finally:
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        if local_json_path and os.path.exists(local_json_path):
            os.remove(local_json_path)
        # Очищаем память после обработки
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def iso8601_to_dd_mm_yyyy(iso_date_str: str) -> str:
    dt = datetime.strptime(iso_date_str, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%d.%m.%Y")

def extract_event_uuid_from_s3_prefix(s3_prefix: str) -> Optional[str]:
    """
    Извлекает UUID события из S3 пути.
    Ожидаемый формат: uuid/timestamp/ или uuid/timestamp
    
    Args:
        s3_prefix: Путь в S3, например "f6c28cf1-4c6c-44b4-a670-35158d9798a1/2025-11-10T11-21-00/"
    
    Returns:
        UUID строкой или None если не удалось извлечь
    """
    try:
        # Убираем trailing slash и разбиваем по /
        clean_prefix = s3_prefix.rstrip('/')
        parts = clean_prefix.split('/')
        
        # UUID должен быть первой частью и иметь длину 36 символов
        if len(parts) >= 1:
            potential_uuid = parts[0]
            
            # Проверяем, что это похоже на UUID (36 символов, содержит дефисы)
            if (len(potential_uuid) == 36 and 
                potential_uuid.count('-') == 4 and
                all(part.isalnum() or part == '' for part in potential_uuid.split('-'))):
                
                logging.info(f"🎯 Извлечен UUID из S3 префикса: {potential_uuid}")
                return potential_uuid
            else:
                logging.warning(f"⚠️ Не удалось извлечь UUID из S3 префикса: {s3_prefix}")
                return None
        else:
            logging.warning(f"⚠️ Неверный формат S3 префикса: {s3_prefix}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Ошибка при извлечении UUID из {s3_prefix}: {e}")
        return None


def send_email_to_owner(owner_email: str, event_uuid: str, result_data: dict, s3_prefix: str, date: str):
    """
    Отправляет результат транскрипции на email владельца события.
    
    Args:
        owner_email: Email владельца события
        event_uuid: UUID события
        result_data: Данные результата обработки
        s3_prefix: Исходный S3 префикс
    """
    try:
       

        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            user_id = segment.get("user_id")

            speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
            if user_id:
                user_stats[user_id] = user_stats.get(user_id, 0) + 1

        # Формируем тему письма
        subject = f"расшифровка dion-конференции за {date}"

        # Формируем тело письма (HTML)
        body = f""""""

        # Добавляем статистику по спикерам
        for speaker, count in sorted(speaker_stats.items(), key=lambda x: x[1], reverse=True):
            user_id = next((seg.get('user_id') for seg in segments if seg.get('speaker') == speaker), None)
            user_info = next((seg.get('user_info') for seg in segments if seg.get('speaker') == speaker), None)

            user_display = ""
            if user_info:
                user_display = f" ({user_info.get('name', user_info.get('email', user_id))})"
            elif user_id:
                user_display = f" (user_id: {user_id})"

            body += f"<li>{speaker}: {count} сегментов{user_display}</li>\n"

        # Добавляем примеры сегментов
        body += f"""
<h3>📝 Примеры транскрипции (первые 5 сегментов):</h3>
"""

        for i, segment in enumerate(segments[:5]):
            speaker = segment.get('speaker', 'unknown')
            text = segment.get('text', '')


            body += f"""
<div style="margin-bottom: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
    <strong> {speaker}: {text}
</div>
"""

        

        body += f"""
<hr/>
<p><em>Обработано системой транскрипции Dion</em></p>
"""

        # Отправляем письмо через общий метод
        send_email(subject=subject, body=body, to_email=owner_email)
        logging.info(f"📧 Результат отправлен на email владельца: {owner_email}")

    except Exception as e:
        logging.error(f"❌ Ошибка при отправке email владельцу {owner_email}: {e}")
        # Резервная отправка
        fallback_subject = f"Транскрипция {s3_prefix}"
        fallback_body = json.dumps(result_data, ensure_ascii=False, indent=2)
        send_email(subject=fallback_subject, body=fallback_body)

# ------------------- Фоновый цикл -------------------
async def background_loop():
    if s3 is None:
        logging.info("S3 отключен или недоступен: фоновый цикл не запускается")
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    """Фоновый цикл для обработки директорий UUID/timestamp с параллельной обработкой"""
    processed_dirs = set()
    executor = ThreadPoolExecutor(max_workers=settings.PARALLEL_WORKERS)
    
    logging.info(f"🚀 Запуск фонового цикла с {settings.PARALLEL_WORKERS} параллельными потоками")
    
    while True:
        try:
            # Получаем список директорий первого уровня (UUID)
            response = s3.list_objects_v2(Bucket=settings.S3_BUCKET, Delimiter='/')
            uuid_prefixes = response.get("CommonPrefixes", [])
            
            #if uuid_prefixes:
            #    logging.info(f"📁 Найдено UUID директорий: {len(uuid_prefixes)}")
            
            # Собираем все директории для обработки
            directories_to_process = []
            
            # Для каждой UUID директории ищем вложенные timestamp директории
            for uuid_prefix_obj in uuid_prefixes:
                uuid_prefix = uuid_prefix_obj["Prefix"]
                
                # Получаем список вложенных директорий (timestamp)
                timestamp_response = s3.list_objects_v2(
                    Bucket=settings.S3_BUCKET, 
                    Prefix=uuid_prefix, 
                    Delimiter='/'
                )
                timestamp_prefixes = timestamp_response.get("CommonPrefixes", [])
                
                if not timestamp_prefixes:
                    #logging.info(f"📭 В UUID директории {uuid_prefix} не найдено timestamp директорий")
                    continue
                
                for timestamp_prefix_obj in timestamp_prefixes:
                    timestamp_prefix = timestamp_prefix_obj["Prefix"]
                    
                    # Пропускаем уже обработанные директории
                    if timestamp_prefix in processed_dirs:
                        logging.info(f"⏭ Пропущена уже обработанная директория: {timestamp_prefix}")
                        continue
                    
                    # Проверяем структуру: UUID/timestamp/
                    parts = timestamp_prefix.rstrip('/').split('/')
                    if len(parts) >= 2:
                        uuid_part = parts[-2]
                        timestamp_part = parts[-1]

                        if uuid_part not in settings.UUID_WHITELIST:
                            #logging.info(f"⚠️ UUID {uuid_part} не в белом списке, пропускаем")
                            continue
                        
                        # Проверяем, что это похоже на UUID и timestamp
                        if len(uuid_part) == 36 and 'T' in timestamp_part:
                            # Проверяем наличие файлов в директории
                            dir_response = s3.list_objects_v2(
                                Bucket=settings.S3_BUCKET, 
                                Prefix=timestamp_prefix
                            )
                            dir_files = dir_response.get("Contents", [])
                            
                            if not dir_files:
                                logging.info(f"📭 Директория {timestamp_prefix} пуста")
                                continue
                            
                            # Проверяем наличие MP4 файла
                            has_mp4 = any(
                                os.path.basename(f["Key"]).lower().endswith('.mp4') 
                                for f in dir_files
                            )
                            
                            if has_mp4:
                                directories_to_process.append(timestamp_prefix)
                                logging.info(f"🎬 Найдена директория для обработки: {timestamp_prefix}")
                            else:
                                logging.info(f"📭 В директории {timestamp_prefix} нет MP4 файла")
                        else:
                            logging.info(f"⚠️ Директория {timestamp_prefix} не соответствует формату UUID/timestamp")
                    else:
                        logging.info(f"⚠️ Неверная структура директории: {timestamp_prefix}")
            
            # Обрабатываем директории параллельно
            if directories_to_process:
                logging.info(f"🎬 Найдено {len(directories_to_process)} директорий для обработки")
                
                # Запускаем обработку в пуле потоков
                futures = []
                for timestamp_prefix in directories_to_process:
                    future = executor.submit(process_directory, timestamp_prefix)
                    futures.append((future, timestamp_prefix))
                
                # Ждем завершения всех задач
                for future, timestamp_prefix in futures:
                    try:
                        future.result(timeout=3600)  # Таймаут 1 час на обработку
                        processed_dirs.add(timestamp_prefix)
                        logging.info(f"✅ Завершена обработка: {timestamp_prefix}")
                    except Exception as e:
                        logging.error(f"❌ Ошибка при обработке {timestamp_prefix}: {e}")
            else:
                # Логируем, почему не найдено директорий для обработки
                if uuid_prefixes:
                    logging.debug(f"ℹ️ Найдено {len(uuid_prefixes)} UUID директорий, но нет новых timestamp директорий для обработки")
                else:
                    logging.debug("ℹ️ Нет UUID директорий для проверки")
        
        except Exception as e:
            logging.error(f"❌ Ошибка в фоновом цикле: {e}")
            import traceback
            logging.error(traceback.format_exc())
            await asyncio.sleep(30)
        
        await asyncio.sleep(settings.CHECK_INTERVAL)

def patch_torch_for_weights_only():
    """Патч для совместимости с PyTorch 2.6+"""
    try:
        # Проверяем версию PyTorch
        torch_version = torch.__version__
        logging.info(f"🔧 PyTorch version: {torch_version}")
        
        # Для версий 2.6 и выше
        if tuple(map(int, torch_version.split('.')[:2])) >= (2, 6):
            logging.info("🎯 Применяем патч для PyTorch 2.6+ (weights_only=False)")
            
            # Переопределяем метод загрузки для безопасной загрузки
            original_load = torch.load
            
            def patched_load(f, map_location=None, pickle_module=None, 
                           weights_only=None, **kwargs):
                # Всегда используем weights_only=False для моделей
                return original_load(f, map_location=map_location, 
                                   pickle_module=pickle_module,
                                   weights_only=False, **kwargs)
            
            torch.load = patched_load
            logging.info("✅ Патч применен успешно")
            
    except Exception as e:
        logging.warning(f"⚠️ Не удалось применить патч для PyTorch: {e}")

@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи при старте"""
    logging.info("🚀 Предзагрузка моделей...")
    preload_all_models()
    if s3 is not None:
        logging.info("🚀 Запуск оптимизированного фонового цикла...")
        asyncio.create_task(background_loop())
    else:
        logging.info("⏭ Фоновый цикл не запущен: S3 отключен или недоступен")

@app.get("/")
def read_root():
    return {"status": "ok", "optimized": True, "version": "2.0"}

@app.get("/health")
def health_check():
    device_type = PerformanceOptimizer.get_available_device()
    return {
        "status": "healthy",
        "device": device_type,
        "models_loaded": diarization_pipeline is not None and whisper_model is not None,
        "parallel_workers": settings.PARALLEL_WORKERS,
        "transcription_mode": settings.TRANSCRIPTION_MODE,
        "whisper_model": settings.WHISPER_MODEL,
        "pyannote_model": settings.PYANNOTE_MODEL,
        "diarization": {
            "clustering_threshold": settings.DIARIZATION_CLUSTERING_THRESHOLD,
            "min_cluster_size": settings.DIARIZATION_MIN_CLUSTER_SIZE,
            "min_duration_off": settings.DIARIZATION_MIN_DURATION_OFF,
            "min_duration_on": settings.DIARIZATION_MIN_DURATION_ON
        }
    }

if __name__ == "__main__":
    if s3 is not None:
        asyncio.run(background_loop())
    else:
        logging.info("S3 отключен или недоступен: выход без запуска фонового цикла")
