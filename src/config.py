"""
Configuración centralizada para el proyecto de clasificación de propinas de taxi NYC.
"""

# Configuración de características
NUMERIC_FEATURES = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "high_tip"
EPS = 1e-7

# Configuración del modelo
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Configuración de datos
DATA_URLS = {
    '2020-01': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet',
    '2020-02': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet',
    '2020-03': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-03.parquet',
    '2020-04': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet',
    '2020-05': 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-05.parquet'
}

# Configuración de umbrales
TIP_THRESHOLD = 0.2  # 20% del costo del viaje
DEFAULT_CONFIDENCE = 0.5

# Configuración de rutas
MODEL_PATH = "./model/random_forest.joblib"
DATA_DIR = "./app/data/"

# Configuración del servidor
SERVER_CONFIG = {
    'host': 'localhost',
    'port': 8000,
    'title': 'Taxi Tip Prediction API'
} 