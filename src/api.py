"""
Módulo para la API FastAPI del modelo de predicción de propinas.
"""

import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from .config import SERVER_CONFIG, FEATURES, DEFAULT_CONFIDENCE
from .predict import cargar_modelo, predecir_viaje_simple
from .features import validate_features

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar nest_asyncio para Jupyter
nest_asyncio.apply()


class TipPredictionRequest(BaseModel):
    """Modelo Pydantic para las solicitudes de predicción."""
    pickup_weekday: float
    pickup_hour: float
    work_hours: float
    pickup_minute: float
    passenger_count: float
    trip_distance: float
    trip_time: float
    trip_speed: float
    PULocationID: float
    DOLocationID: float
    RatecodeID: float


class TipPredictionResponse(BaseModel):
    """Modelo Pydantic para las respuestas de predicción."""
    prediction: int
    probability: float
    confidence_threshold: float
    high_tip: bool


class BatchPredictionRequest(BaseModel):
    """Modelo Pydantic para predicciones en lote."""
    features: List[List[float]]
    confidence: Optional[float] = DEFAULT_CONFIDENCE


class BatchPredictionResponse(BaseModel):
    """Modelo Pydantic para respuestas de predicciones en lote."""
    predictions: List[int]
    probabilities: List[float]
    confidence_threshold: float


class ModelInfoResponse(BaseModel):
    """Modelo Pydantic para información del modelo."""
    model_type: str
    features: List[str]
    n_features: int
    confidence_threshold: float


class TaxiTipAPI:
    """Clase principal para la API de predicción de propinas."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa la API.
        
        Args:
            model_path: Ruta del modelo (opcional)
        """
        self.app = FastAPI(title=SERVER_CONFIG['title'])
        self.model = None
        self.model_path = model_path
        
        # Configurar endpoints
        self._setup_endpoints()
        
    def _setup_endpoints(self):
        """Configura todos los endpoints de la API."""
        
        @self.app.get("/")
        async def root():
            """Endpoint raíz con información básica."""
            return {
                "message": "Taxi Tip Prediction API",
                "version": "1.0.0",
                "endpoints": [
                    "/predict",
                    "/predict_batch", 
                    "/model_info",
                    "/health"
                ]
            }
        
        @self.app.post("/predict", response_model=TipPredictionResponse)
        async def predict_tip(request: TipPredictionRequest, confidence: float = DEFAULT_CONFIDENCE):
            """
            Predice si un viaje tendrá propina alta.
            
            Args:
                request: Características del viaje
                confidence: Umbral de confianza
                
            Returns:
                Predicción y probabilidad
            """
            try:
                # Cargar modelo si no está cargado
                if self.model is None:
                    self.model = cargar_modelo(self.model_path)
                
                # Convertir request a diccionario
                features_dict = request.dict()
                
                # Realizar predicción
                result = predecir_viaje_simple(self.model, features_dict, confidence)
                
                return TipPredictionResponse(**result)
                
            except Exception as e:
                logger.error(f"Error en predicción: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest):
            """
            Predice múltiples viajes en lote.
            
            Args:
                request: Lista de características de viajes
                
            Returns:
                Lista de predicciones y probabilidades
            """
            try:
                # Cargar modelo si no está cargado
                if self.model is None:
                    self.model = cargar_modelo(self.model_path)
                
                features_array = np.array(request.features)
                
                # Validar características
                if not validate_features(features_array):
                    raise ValueError("Formato de características inválido")
                
                # Realizar predicciones
                probas = self.model.predict_proba(features_array)
                predictions = [1 if p[1] >= request.confidence else 0 for p in probas]
                probabilities = [p[1] for p in probas]
                
                return BatchPredictionResponse(
                    predictions=predictions,
                    probabilities=probabilities,
                    confidence_threshold=request.confidence
                )
                
            except Exception as e:
                logger.error(f"Error en predicción en lote: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model_info", response_model=ModelInfoResponse)
        async def get_model_info():
            """
            Obtiene información sobre el modelo.
            
            Returns:
                Información del modelo
            """
            try:
                # Cargar modelo si no está cargado
                if self.model is None:
                    self.model = cargar_modelo(self.model_path)
                
                return ModelInfoResponse(
                    model_type=type(self.model).__name__,
                    features=FEATURES,
                    n_features=len(FEATURES),
                    confidence_threshold=DEFAULT_CONFIDENCE
                )
                
            except Exception as e:
                logger.error(f"Error obteniendo información del modelo: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """
            Verifica el estado de salud de la API.
            
            Returns:
                Estado de salud
            """
            try:
                # Verificar que el modelo esté cargado
                if self.model is None:
                    self.model = cargar_modelo(self.model_path)
                
                return {
                    "status": "healthy",
                    "model_loaded": self.model is not None,
                    "features_count": len(FEATURES)
                }
                
            except Exception as e:
                logger.error(f"Error en health check: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }
    
    def run(self, host: str = None, port: int = None):
        """
        Ejecuta el servidor FastAPI.
        
        Args:
            host: Host del servidor
            port: Puerto del servidor
        """
        host = host or SERVER_CONFIG['host']
        port = port or SERVER_CONFIG['port']
        
        logger.info(f"Iniciando servidor en {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_api_app(model_path: str = None) -> FastAPI:
    """
    Crea una instancia de la aplicación FastAPI.
    
    Args:
        model_path: Ruta del modelo
        
    Returns:
        Aplicación FastAPI
    """
    api = TaxiTipAPI(model_path)
    return api.app


def run_server(model_path: str = None, host: str = None, port: int = None):
    """
    Ejecuta el servidor de la API.
    
    Args:
        model_path: Ruta del modelo
        host: Host del servidor
        port: Puerto del servidor
    """
    api = TaxiTipAPI(model_path)
    api.run(host, port)


# Función de conveniencia para predicción simple
def predict_taxi_trip(features_trip: np.ndarray, confidence: float = DEFAULT_CONFIDENCE) -> int:
    """
    Predice si un viaje tendrá propina alta (función de conveniencia).
    
    Args:
        features_trip: Array con características del viaje
        confidence: Umbral de confianza
        
    Returns:
        1 si propina alta, 0 en caso contrario
    """
    try:
        # Cargar modelo
        model = cargar_modelo()
        
        # Validar características
        if not validate_features(features_trip.reshape(1, -1)):
            raise ValueError("Formato de características inválido")
        
        # Realizar predicción
        pred_value = model.predict_proba(features_trip.reshape(1, -1))[0][1]
        
        return 1 if pred_value >= confidence else 0
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise


if __name__ == "__main__":
    # Ejecutar servidor
    run_server() 