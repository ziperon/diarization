import os


S3_BUCKET = "dionrecord"
S3_ENABLED = os.getenv("S3_ENABLED", "1") == "1"
LOCAL_TMP = "./tmp/audiot"
CHECK_INTERVAL = 10
SUPPORTED_EXT = ['mp3', 'm4a', 'wav', 'flac']
CHUNK_MS = 30_000      # 30 seconds
OVERLAP_MS = 1_000     # 1 second
MIN_SPEECH_MS = 300    # отрезаем микротишину
# Количество параллельных потоков для обработки диаризаций
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "1"))  # По умолчанию 2 потока

EMAIL_HOST = "cmsk01mx03.corp.vtbcapital.internal"
EMAIL_PORT = 25
EMAIL_USER = "CORP\\svc_transcribation"
EMAIL_PASS = "gAAAAABpHWTiaF-wM7YHb99cv_41rhRSaASCbDHqZLKUHa7inJLXS_YCYrvGrQvFNhfvBMt8kpO-_SJB5QT9Cm6v1MjGgv7-5c7M2sw3dFtI0in8aF7zM3c="
EMAIL_TO = ""
EMAIL_FROM = "svc_transcribation@mncap.ru"
EMAIL_USE_TLS = True

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] =""
os.environ["HF_TOKEN"]="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"

# DION API конфигурация
DION_ACCESS_TOKEN = "gAAAAABpHWRF9tF1OtexVslUyAvsP8C9dAefpv3o6yBYkq8Za-qe0jho5PuAR3SOSwMROFQvpKE-qwKWzYOuKrbxAfmAD7d53HI5hq20-I6JuGq_DwTnQcYLSDVOlrdRLh4NNvYMVvm1H7ZHXnKGRlUn5-NqgrNlbPG90IKTGaAEaGquXSeYy1IjITn_EOrYKkqKrexU-F7-" #os.getenv("DION_ACCESS_TOKEN")  # Токен из переменной окружения
DION_API_ENABLED = bool(DION_ACCESS_TOKEN)  # Включаем только если токен задан

MODELS_DIR = "./models"
# Модель диаризации: 
# - "pyannote/speaker-diarization-3.1" (стабильная, по умолчанию)
# - "pyannote/speaker-diarization-community-1" (лучше качество, новее)
# - "pyannote/speaker-diarization-precision-2" (премиум, требует API ключ)
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
WHISPER_MODEL = "large-v3"  # ← ИСПОЛЬЗУЕМ large-v2 (быстрее чем v3)
# Режим качества транскрипции: "fast" (скорость) или "quality" (качество)
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "quality").lower()  # По умолчанию качество
# Основной ASR и параметры GigaAM
GIGAAM_MODEL_TYPE = os.getenv("GIGAAM_MODEL_TYPE", "v3_e2e_rnnt")
ASR_PRIMARY = os.getenv("ASR_PRIMARY", "gigaam")  # "gigaam" | "whisper"

# Параметры диаризации для улучшения разделения спикеров
# Порог кластеризации: меньше значение = больше спикеров (по умолчанию 0.704, для лучшего разделения используйте 0.5-0.65)
DIARIZATION_CLUSTERING_THRESHOLD = float(os.getenv("DIARIZATION_CLUSTERING_THRESHOLD", "0.704"))
# Минимальный размер кластера: меньше значение = больше спикеров (по умолчанию 12, для лучшего разделения используйте 6-10)
DIARIZATION_MIN_CLUSTER_SIZE = int(os.getenv("DIARIZATION_MIN_CLUSTER_SIZE", "1"))
# Минимальная длительность паузы между спикерами (секунды)
DIARIZATION_MIN_DURATION_OFF = float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.1"))
# Минимальная длительность сегмента речи (секунды)
DIARIZATION_MIN_DURATION_ON = float(os.getenv("DIARIZATION_MIN_DURATION_ON", "0.1"))

ASR_SEGMENT_BY_DIARIZATION = int(os.getenv("ASR_SEGMENT_BY_DIARIZATION", "0"))  # 1 = распознавать по сегментам диаризации



AWS_ACCESS_KEY_ID='gAAAAABpHWPuyFdQDEh5-66S4lrrQ5dUlgqmD-VrQd3X0EzBXZE8MVk6gq_adFx_bNqKboF6Och3yvogxI3hM0daNN3xfptWPVGOjEYmTlrOOeeMDQVLVEc=',
AWS_SECRET_ACCESS_KEY='gAAAAABpHWQKDgahGLCsjIyW8lj2NBKJuw6FeeopB91Ui8FLiDDkxL2iI-8Xankn8uMc623Hs1jcMTIF3mx4Slm3Mr7b1xjGXQGPoOs2AO9Ea-SjYFDEvZgiNvQoiEs6tQMyn4BFd7q8',

S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# Параметры привязки user_id по времени
# Расширение окна сегмента при поиске пересечений с треками (мс)
USER_MATCH_EXPAND_MS = int(os.getenv("USER_MATCH_EXPAND_MS", "3000"))
# Ширина контекстного окна до/после сегмента для доминирующего пользователя (мс)
USER_CONTEXT_WINDOW_MS = int(os.getenv("USER_CONTEXT_WINDOW_MS", "5000"))
# Порог минимального перекрытия сегмента с треком (доля 0..1)
MIN_OVERLAP_PCT_SEGMENT = float(os.getenv("MIN_OVERLAP_PCT_SEGMENT", "0.1"))
# Порог «близости по началу» для контекстной эвристики (мс)
USER_NEAR_START_DIFF_MS = int(os.getenv("USER_NEAR_START_DIFF_MS", "2000"))

# белый список доступных к обработке
UUID_WHITELIST = [
    "f6c28cf1-4c6c-44b4-a670-35158d9798a0", #https://dion.vc/event/nikolay-kozyrev
    "29b19648-a5d8-4b71-be6b-9c5f515b7920", #https://dion.vc/event/ciooffice-alexeypavlin
    "a7aefb8a-f356-4c17-84fa-f9b84adfed02", #https://dion.vc/event/ciooffice-it1
    "04e6b432-961f-44a3-9e5d-fc967aefa9a7", #https://dion.vc/event/ciooffice-it2
    "5d75a776-9ab7-4b2c-8be6-21cba696e672", #https://dion.vc/event/ciooffice-it3
    "d8794e89-a0cc-400b-bc3a-51b5715171c1", #https://dion.vc/event/ciooffice-itdep
    "3f704ebc-91b8-4d2b-9000-0642b9863995", #https://dion.vc/event/anton-orlov
    "355f027c-6aa6-4d91-b687-10cd56df5954", #https://dion.vc/event/ciooffice-sergeybelousov
    "ea0230d6-2344-486c-b6f9-acbc55fcc636", #https://dion.vc/event/olga-senyushkina
    "82e09417-a9c9-46fe-b534-46bae14b23ef",
    "9fcf1a91-3c89-437e-ae9d-6a4e9d1a8f6d" #https://dion.vc/event/natalia-solodova
]
