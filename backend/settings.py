import os


S3_BUCKET = "diarization-files"
LOCAL_TMP = os.path.join(os.getcwd(), "tmp", "audiot")
CHECK_INTERVAL = 10
SUPPORTED_EXT = ['mp3', 'm4a', 'wav', 'flac']
# Количество параллельных потоков для обработки диаризаций
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "1"))  # По умолчанию 2 потока

EMAIL_HOST = "smtp.mailmug.net"
EMAIL_PORT = 2525
EMAIL_USER = "3o3pxmzymuwlbhwf"
EMAIL_PASS = "aiojg5clllgevovm"
EMAIL_TO = "your-email@gmail.com"
EMAIL_FROM = "your-email@gmail.com"
EMAIL_USE_TLS = False

# DION API конфигурация
DION_ACCESS_TOKEN = os.getenv("DION_ACCESS_TOKEN")  # Токен из переменной окружения
DION_API_ENABLED = bool(DION_ACCESS_TOKEN)  # Включаем только если токен задан

MODELS_DIR = "./models"
# Модель диаризации: 
# - "pyannote/speaker-diarization-3.1" (стабильная, по умолчанию)
# - "pyannote/speaker-diarization-community-1" (лучше качество, новее)
# - "pyannote/speaker-diarization-precision-2" (премиум, требует API ключ)
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
WHISPER_MODEL = "large-v2"  # ← ИСПОЛЬЗУЕМ large-v2 (быстрее чем v3)
# Режим качества транскрипции: "fast" (скорость) или "quality" (качество)
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "quality").lower()  # По умолчанию качество

# Параметры диаризации для улучшения разделения спикеров
# Порог кластеризации: меньше значение = больше спикеров (по умолчанию 0.704, для лучшего разделения используйте 0.5-0.65)
DIARIZATION_CLUSTERING_THRESHOLD = float(os.getenv("DIARIZATION_CLUSTERING_THRESHOLD", "0.704"))
# Минимальный размер кластера: меньше значение = больше спикеров (по умолчанию 12, для лучшего разделения используйте 6-10)
DIARIZATION_MIN_CLUSTER_SIZE = int(os.getenv("DIARIZATION_MIN_CLUSTER_SIZE", "12"))
# Минимальная длительность паузы между спикерами (секунды)
DIARIZATION_MIN_DURATION_OFF = float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.0"))
# Минимальная длительность сегмента речи (секунды)
DIARIZATION_MIN_DURATION_ON = float(os.getenv("DIARIZATION_MIN_DURATION_ON", "0.0"))