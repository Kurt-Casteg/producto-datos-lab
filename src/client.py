"""
Módulo cliente para interactuar con la API de predicción de propinas.
"""

import requests
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
import logging

from .config import SERVER_CONFIG, FEATURES, DEFAULT_CONFIDENCE

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiTipClient:
    """Cliente para interactuar con la API de predicción de propinas."""
    
    def __init__(self, base_url: str = None):
        """
        Inicializa el cliente.
        
        Args:
            base_url: URL base del servidor
        """
        self.base_url = base_url or f"http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}"
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> requests.Response:
        """
        Realiza una petición HTTP.
        
        Args:
            endpoint: Endpoint de la API
            method: Método HTTP
            data: Datos a enviar
            
        Returns:
            Respuesta de la petición
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición HTTP: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del servidor.
        
        Returns:
            Estado de salud del servidor
        """
        response = self._make_request("/health")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo.
        
        Returns:
            Información del modelo
        """
        response = self._make_request("/model_info")
        return response.json()
    
    def predict_single(
        self, 
        features: Dict[str, float], 
        confidence: float = DEFAULT_CONFIDENCE
    ) -> Dict[str, Any]:
        """
        Predice si un viaje tendrá propina alta.
        
        Args:
            features: Diccionario con características del viaje
            confidence: Umbral de confianza
            
        Returns:
            Predicción y probabilidad
        """
        # Validar características
        missing_features = set(FEATURES) - set(features.keys())
        if missing_features:
            raise ValueError(f"Características faltantes: {missing_features}")
        
        # Preparar datos
        data = {
            **features,
            "confidence": confidence
        }
        
        response = self._make_request("/predict", method="POST", data=data)
        return response.json()
    
    def predict_batch(
        self, 
        features_list: List[List[float]], 
        confidence: float = DEFAULT_CONFIDENCE
    ) -> Dict[str, Any]:
        """
        Predice múltiples viajes en lote.
        
        Args:
            features_list: Lista de características de viajes
            confidence: Umbral de confianza
            
        Returns:
            Lista de predicciones y probabilidades
        """
        # Validar formato
        if not features_list:
            raise ValueError("Lista de características vacía")
        
        for i, features in enumerate(features_list):
            if len(features) != len(FEATURES):
                raise ValueError(f"Características en posición {i} tienen longitud incorrecta")
        
        # Preparar datos
        data = {
            "features": features_list,
            "confidence": confidence
        }
        
        response = self._make_request("/predict_batch", method="POST", data=data)
        return response.json()
    
    def predict_from_array(
        self, 
        features_array: np.ndarray, 
        confidence: float = DEFAULT_CONFIDENCE
    ) -> Dict[str, Any]:
        """
        Predice desde un array de características.
        
        Args:
            features_array: Array con características
            confidence: Umbral de confianza
            
        Returns:
            Predicción y probabilidad
        """
        if features_array.ndim == 1:
            # Predicción individual
            features_dict = dict(zip(FEATURES, features_array))
            return self.predict_single(features_dict, confidence)
        elif features_array.ndim == 2:
            # Predicción en lote
            features_list = features_array.tolist()
            return self.predict_batch(features_list, confidence)
        else:
            raise ValueError("Array debe ser 1D (individual) o 2D (lote)")
    
    def test_connection(self) -> bool:
        """
        Prueba la conexión con el servidor.
        
        Returns:
            True si la conexión es exitosa
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Error probando conexión: {e}")
            return False


def response_from_server(
    url: str, 
    item_features: Dict[str, float], 
    verbose: bool = True
) -> requests.Response:
    """
    Hace una solicitud POST al servidor y retorna la respuesta.
    
    Args:
        url: URL a la que se envía la consulta
        item_features: Diccionario con características del viaje
        verbose: Si imprimir información de estado
        
    Returns:
        Respuesta del servidor
    """
    try:
        response = requests.post(url, json=item_features)
        status_code = response.status_code
        
        if verbose:
            msg = "¡Todo funcionó bien!" if status_code == 200 else "Hubo un error al ejecutar la solicitud."
            print(msg)
        
        return response
        
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Error en la petición: {e}")
        raise


def create_sample_features() -> Dict[str, float]:
    """
    Crea características de ejemplo para un viaje.
    
    Returns:
        Diccionario con características de ejemplo
    """
    return {
        "pickup_weekday": 5.0,
        "pickup_hour": 0.0,
        "work_hours": 0.0,
        "pickup_minute": 17.0,
        "passenger_count": 1.0,
        "trip_distance": 2.599,
        "trip_time": 777.0,
        "trip_speed": 3.3462034e-03,
        "PULocationID": 145.0,
        "DOLocationID": 7.0,
        "RatecodeID": 1.0
    }


def test_client_workflow():
    """
    Prueba el flujo completo del cliente.
    """
    logger.info("Iniciando prueba del cliente")
    
    # Crear cliente
    client = TaxiTipClient()
    
    # Verificar conexión
    if not client.test_connection():
        logger.error("No se pudo conectar al servidor")
        return
    
    logger.info("Conexión exitosa con el servidor")
    
    # Obtener información del modelo
    try:
        model_info = client.get_model_info()
        logger.info(f"Modelo: {model_info['model_type']}")
        logger.info(f"Características: {model_info['n_features']}")
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {e}")
    
    # Probar predicción individual
    try:
        features = create_sample_features()
        prediction = client.predict_single(features)
        logger.info(f"Predicción individual: {prediction}")
    except Exception as e:
        logger.error(f"Error en predicción individual: {e}")
    
    # Probar predicción en lote
    try:
        features_batch = [list(features.values()) for _ in range(3)]
        predictions_batch = client.predict_batch(features_batch)
        logger.info(f"Predicciones en lote: {len(predictions_batch['predictions'])} predicciones")
    except Exception as e:
        logger.error(f"Error en predicción en lote: {e}")
    
    logger.info("Prueba del cliente completada")


if __name__ == "__main__":
    # Ejecutar prueba del cliente
    test_client_workflow() 