# 🚕 Taxi Tip Prediction - Laboratorio de Machine Learning

Este proyecto implementa un modelo de Machine Learning para predecir si un pasajero de taxi en NYC dejará una propina alta (≥20% del costo del viaje). El proyecto incluye entrenamiento de modelos, API REST, monitoreo y CI/CD con GitHub Actions.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Ejecución del Proyecto](#ejecución-del-proyecto)
- [Uso de la API](#uso-de-la-api)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Testing y CI/CD](#testing-y-cicd)
- [Notebooks de Desarrollo](#notebooks-de-desarrollo)

## 🎯 Descripción del Proyecto

Este laboratorio está inspirado en la unidad 1 del curso [Introduction to Machine Learning in Production (DeepLearning.AI)](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/home/welcome) y implementa:

- **Modelo de Clasificación**: Random Forest para predecir propinas altas
- **API REST**: FastAPI para servir predicciones
- **Monitoreo**: Sistema de monitoreo de rendimiento del modelo
- **CI/CD**: Pipeline automatizado con GitHub Actions
- **Testing**: Tests unitarios con pytest

## 📁 Estructura del Proyecto

```
producto-datos-lab/
├── 📁 app/                    # Aplicación principal
│   ├── main.py               # Servidor FastAPI simple
│   ├── test_rfc.py          # Tests unitarios
│   ├── 📁 data/             # Datos de prueba
│   └── 📁 model/            # Modelos entrenados
├── 📁 src/                   # Código modular
│   ├── api.py               # API FastAPI modular
│   ├── train.py             # Entrenamiento de modelos
│   ├── predict.py           # Predicciones
│   ├── features.py          # Ingeniería de características
│   ├── monitoring.py        # Monitoreo
│   ├── client.py            # Cliente para API
│   └── config.py            # Configuración
├── 📁 notebooks/            # Jupyter notebooks
├── 📁 model/                # Modelos guardados
├── 📁 docs/                 # Documentación
├── requirements.txt          # Dependencias Python
└── README.md               # Este archivo
```

## 🔧 Requisitos Previos

### Opción 1: Conda (Recomendado)
- [Conda](https://docs.conda.io/en/latest/) instalado en tu sistema

### Opción 2: uv (Alternativa moderna)
- [uv](https://docs.astral.sh/uv/) instalado en tu sistema

### Opción 3: Python virtual environment
- Python 3.8+ instalado
- pip para gestión de paquetes

## 🚀 Instalación

### Método 1: Usando Conda (Recomendado)

1. **Crear entorno virtual:**
```bash
conda create --name producto-datos-lab python=3.9
conda activate producto-datos-lab
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Configurar Jupyter:**
```bash
python -m ipykernel install --user --name producto-datos-lab
```

### Método 2: Usando uv (Más rápido)

1. **Crear entorno virtual:**
```bash
uv venv
```

2. **Activar entorno:**
```bash
# Windows
producto-datos-lab\Scripts\activate

# macOS/Linux
source producto-datos-lab/bin/activate
```

3. **Instalar dependencias:**
```bash
uv pip install -r requirements.txt
```

4. **Configurar Jupyter:**
```bash
python -m ipykernel install --user --name producto-datos-lab
```

### Método 3: Python virtual environment

1. **Crear entorno virtual:**
```bash
python -m venv producto-datos-lab
```

2. **Activar entorno:**
```bash
# Windows
producto-datos-lab\Scripts\activate

# macOS/Linux
source producto-datos-lab/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Ejecución del Proyecto

### 1. Entrenamiento del Modelo

#### Opción A: Usando el código modular (src/)

```bash
# Desde la raíz del proyecto
python -c "
from src.train import pipeline_entrenamiento_completo
from src.dataset import cargar_datos_nyc
import pandas as pd

# Cargar datos
df = cargar_datos_nyc('2020-01')  # Enero 2020
modelo, metrics = pipeline_entrenamiento_completo(df)
print(f'Modelo entrenado con F1-Score: {metrics[\"f1_score\"]:.4f}')
"
```

#### Opción B: Usando notebooks

```bash
jupyter lab
```

Luego ejecuta el notebook `00_nyc-taxi-model.ipynb` para entrenar el modelo.

### 2. Ejecutar la API

#### Opción A: API Simple (app/main.py)

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Opción B: API Modular (src/api.py)

```bash
python -c "
from src.api import run_server
run_server()
"
```

### 3. Probar la API

#### Usando curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "pickup_weekday": 1.0,
       "pickup_hour": 14.0,
       "work_hours": 1.0,
       "pickup_minute": 30.0,
       "passenger_count": 2.0,
       "trip_distance": 5.2,
       "trip_time": 1200.0,
       "trip_speed": 15.6,
       "PULocationID": 1.0,
       "DOLocationID": 2.0,
       "RatecodeID": 1.0
     }' \
     -G --data-urlencode "confidence=0.5"
```

#### Usando Python:

```bash
python -c "
from src.client import TaxiTipClient
client = TaxiTipClient('http://localhost:8000')
result = client.predict_tip({
    'pickup_weekday': 1.0,
    'pickup_hour': 14.0,
    'work_hours': 1.0,
    'pickup_minute': 30.0,
    'passenger_count': 2.0,
    'trip_distance': 5.2,
    'trip_time': 1200.0,
    'trip_speed': 15.6,
    'PULocationID': 1.0,
    'DOLocationID': 2.0,
    'RatecodeID': 1.0
}, confidence=0.5)
print(f'Predicción: {result}')
"
```

## 📊 Uso de la API

### Endpoints Disponibles

#### 1. Información del Modelo
```bash
GET http://localhost:8000/model_info
```

#### 2. Predicción Individual
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "pickup_weekday": 1.0,
  "pickup_hour": 14.0,
  "work_hours": 1.0,
  "pickup_minute": 30.0,
  "passenger_count": 2.0,
  "trip_distance": 5.2,
  "trip_time": 1200.0,
  "trip_speed": 15.6,
  "PULocationID": 1.0,
  "DOLocationID": 2.0,
  "RatecodeID": 1.0
}
```

#### 3. Predicción en Lote
```bash
POST http://localhost:8000/predict_batch
Content-Type: application/json

{
  "features": [
    [1.0, 14.0, 1.0, 30.0, 2.0, 5.2, 1200.0, 15.6, 1.0, 2.0, 1.0],
    [2.0, 9.0, 1.0, 15.0, 1.0, 3.1, 900.0, 12.4, 3.0, 4.0, 1.0]
  ],
  "confidence": 0.5
}
```

#### 4. Estado de Salud
```bash
GET http://localhost:8000/health
```

### Documentación Interactiva

Una vez que la API esté ejecutándose, visita:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧠 Entrenamiento de Modelos

### Características del Modelo

El modelo utiliza las siguientes características:

**Numéricas:**
- `pickup_weekday`: Día de la semana (0-6)
- `pickup_hour`: Hora del día (0-23)
- `work_hours`: Si es hora laboral (0/1)
- `pickup_minute`: Minuto del día (0-59)
- `passenger_count`: Número de pasajeros
- `trip_distance`: Distancia del viaje (millas)
- `trip_time`: Tiempo del viaje (segundos)
- `trip_speed`: Velocidad promedio (mph)

**Categóricas:**
- `PULocationID`: ID de ubicación de recogida
- `DOLocationID`: ID de ubicación de destino
- `RatecodeID`: Código de tarifa

### Configuración del Modelo

```python
MODEL_CONFIG = {
    'n_estimators': 100,    # Número de árboles
    'max_depth': 10,        # Profundidad máxima
    'random_state': 42      # Semilla para reproducibilidad
}
```

## 🧪 Testing y CI/CD

### Ejecutar Tests Localmente

```bash
cd app
pytest test_rfc.py -v
```

### Pipeline CI/CD con GitHub Actions

El proyecto incluye un pipeline automatizado que:

1. **Se activa** cuando se hace push a archivos en `app/`
2. **Instala** dependencias
3. **Ejecuta** tests unitarios
4. **Reporta** resultados

Para activar el pipeline:

1. **Fork** el repositorio
2. **Habilita** GitHub Actions en tu fork
3. **Haz cambios** en archivos de `app/`
4. **Push** los cambios

### Estructura del Pipeline

```yaml
name: PD-MDS-Lab
on:
  push:
    paths:
      - 'app/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8.2'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Test with pytest
        run: |
          cd app/
          pytest
```

## 📓 Notebooks de Desarrollo

### Notebooks Disponibles

1. **`00_nyc-taxi-model.ipynb`**: Entrenamiento completo del modelo
2. **`01_server.ipynb`**: Configuración y ejecución del servidor
3. **`02_client.ipynb`**: Cliente para consumir la API
4. **`03_nyc-taxi-model_future_predictions.ipynb`**: Predicciones futuras

### Ejecutar Jupyter Lab

```bash
jupyter lab
```

## 🔍 Monitoreo

El proyecto incluye un sistema de monitoreo que:

- **Rastrea** métricas de rendimiento
- **Detecta** drift en los datos
- **Alerta** sobre degradación del modelo
- **Genera** reportes automáticos

### Ejecutar Monitoreo

```bash
python -c "
from src.monitoring import TaxiTipMonitor
monitor = TaxiTipMonitor()
monitor.run_monitoring()
"
```

## 🛠️ Solución de Problemas

### Error: "Modelo no encontrado"
```bash
# Asegúrate de que el modelo esté entrenado
python -c "from src.train import pipeline_entrenamiento_completo; from src.dataset import cargar_datos_nyc; df = cargar_datos_nyc('2020-01'); pipeline_entrenamiento_completo(df)"
```

### Error: "Puerto 8000 ocupado"
```bash
# Usa un puerto diferente
uvicorn main:app --reload --port 8001
```

### Error: "Dependencias no encontradas"
```bash
# Reinstala las dependencias
pip install -r requirements.txt --force-reinstall
```

## 📈 Métricas de Rendimiento

El modelo típicamente alcanza:
- **F1-Score**: > 0.7
- **Accuracy**: > 0.75
- **Precision**: > 0.8
- **Recall**: > 0.6

## 🤝 Contribución

1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Haz** tus cambios
4. **Ejecuta** los tests
5. **Push** tus cambios
6. **Crea** un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [DeepLearning.AI](https://www.deeplearning.ai/) por el curso de ML in Production
- [Shreya Shankar](https://github.com/shreyashankar/debugging-ml-talk) por el código base
- [GitHub Actions](https://github.com/features/actions) por la automatización de CI/CD

---

**¡Disfruta explorando el mundo del Machine Learning en Producción! 🚀**

