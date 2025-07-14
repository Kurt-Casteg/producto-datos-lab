"""
Módulo para predicciones y evaluación de modelos.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Union, Optional
import logging

from .config import MODEL_PATH, FEATURES, TARGET_COL, DEFAULT_CONFIDENCE
from .features import build_features, validate_features

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_modelo(ruta_modelo: str = MODEL_PATH):
    """
    Carga un modelo serializado con joblib.
    
    Args:
        ruta_modelo: Ruta del modelo a cargar
        
    Returns:
        Modelo cargado
    """
    try:
        modelo = joblib.load(ruta_modelo)
        logger.info(f"Modelo cargado desde {ruta_modelo}")
        return modelo
    except FileNotFoundError:
        logger.error(f"No se encontró el modelo en {ruta_modelo}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise


def predecir(
    modelo, 
    X: Union[pd.DataFrame, np.ndarray],
    confidence: float = DEFAULT_CONFIDENCE
) -> List[int]:
    """
    Realiza predicciones de probabilidad y devuelve las etiquetas predichas.
    
    Args:
        modelo: Modelo entrenado
        X: Características para predecir
        confidence: Umbral de confianza para clasificación
        
    Returns:
        Lista de etiquetas predichas
    """
    # Validar entrada
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if not validate_features(X_array):
        raise ValueError("Formato de características inválido")
    
    # Realizar predicciones
    probas = modelo.predict_proba(X_array)
    etiquetas = [1 if p[1] >= confidence else 0 for p in probas]
    
    logger.info(f"Predicciones realizadas para {len(etiquetas)} muestras")
    
    return etiquetas


def predecir_proba(
    modelo, 
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Realiza predicciones de probabilidad.
    
    Args:
        modelo: Modelo entrenado
        X: Características para predecir
        
    Returns:
        Array con probabilidades de clase positiva
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if not validate_features(X_array):
        raise ValueError("Formato de características inválido")
    
    probas = modelo.predict_proba(X_array)
    return probas[:, 1]  # Probabilidad de clase positiva


def evaluar(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """
    Calcula el F1-score.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        F1-score
    """
    f1 = f1_score(y_true, y_pred)
    logger.info(f"F1-score: {f1:.4f}")
    return f1


def evaluar_completo(
    y_true: Union[List, np.ndarray], 
    y_pred: Union[List, np.ndarray],
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluación completa del modelo con múltiples métricas.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        y_pred_proba: Probabilidades predichas (opcional)
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Métricas básicas
    f1 = f1_score(y_true, y_pred)
    
    # Reporte de clasificación
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'f1_score': f1,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'accuracy': report['accuracy'],
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def predecir_viaje_simple(
    modelo,
    features_dict: Dict[str, float],
    confidence: float = DEFAULT_CONFIDENCE
) -> Dict[str, Any]:
    """
    Predice si un viaje tendrá propina alta basado en un diccionario de características.
    
    Args:
        modelo: Modelo entrenado
        features_dict: Diccionario con características del viaje
        confidence: Umbral de confianza
        
    Returns:
        Diccionario con predicción y probabilidad
    """
    # Convertir diccionario a array
    features_array = np.array([features_dict[feature] for feature in FEATURES])
    
    # Validar características
    if not validate_features(features_array.reshape(1, -1)):
        raise ValueError("Características inválidas")
    
    # Realizar predicción
    proba = modelo.predict_proba(features_array.reshape(1, -1))[0][1]
    prediction = 1 if proba >= confidence else 0
    
    return {
        'prediction': prediction,
        'probability': proba,
        'confidence_threshold': confidence,
        'high_tip': bool(prediction)
    }


def pipeline_prediccion_completo(
    df_raw: pd.DataFrame,
    modelo_path: str = MODEL_PATH,
    confidence: float = DEFAULT_CONFIDENCE
) -> Dict[str, Any]:
    """
    Pipeline completo de predicción desde datos crudos.
    
    Args:
        df_raw: DataFrame con datos crudos
        modelo_path: Ruta del modelo
        confidence: Umbral de confianza
        
    Returns:
        Diccionario con predicciones y métricas
    """
    logger.info("Iniciando pipeline de predicción completo")
    
    # Cargar modelo
    modelo = cargar_modelo(modelo_path)
    
    # Preprocesar datos
    df_proc = build_features(df_raw)
    logger.info(f"Datos preprocesados: {len(df_proc)} muestras")
    
    # Realizar predicciones
    X = df_proc[FEATURES]
    y_true = df_proc[TARGET_COL]
    
    y_pred = predecir(modelo, X, confidence)
    y_pred_proba = predecir_proba(modelo, X)
    
    # Evaluar
    metrics = evaluar_completo(y_true, y_pred, y_pred_proba)
    
    # Preparar resultados
    results = {
        'predictions': y_pred,
        'probabilities': y_pred_proba.tolist(),
        'true_values': y_true.tolist(),
        'metrics': metrics,
        'n_samples': len(df_proc)
    }
    
    logger.info("Pipeline de predicción completado")
    
    return results


def monitorear_rendimiento_temporal(
    modelo_path: str = MODEL_PATH,
    meses: List[str] = None
) -> Dict[str, Any]:
    """
    Monitorea el rendimiento del modelo a lo largo del tiempo.
    
    Args:
        modelo_path: Ruta del modelo
        meses: Lista de meses a evaluar
        
    Returns:
        Diccionario con rendimiento por mes
    """
    from .config import DATA_URLS
    
    if meses is None:
        meses = list(DATA_URLS.keys())
    
    modelo = cargar_modelo(modelo_path)
    resultados = {}
    
    for mes in meses:
        try:
            # Cargar datos del mes
            df = pd.read_parquet(DATA_URLS[mes])
            df_proc = build_features(df.head(1000))  # Muestra para eficiencia
            
            # Evaluar
            X = df_proc[FEATURES]
            y_true = df_proc[TARGET_COL]
            y_pred = predecir(modelo, X)
            
            f1 = f1_score(y_true, y_pred)
            resultados[mes] = {
                'f1_score': f1,
                'n_samples': len(df_proc)
            }
            
            logger.info(f"{mes}: F1-Score = {f1:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluando {mes}: {e}")
            resultados[mes] = {'error': str(e)}
    
    return resultados


def evaluar_meses_tabla(
    modelo_path: str = MODEL_PATH,
    meses: list = None,
    n_muestras: int = 1000
) -> pd.DataFrame:
    """
    Evalúa el modelo para varios meses y devuelve una tabla con los resultados.
    
    Args:
        modelo_path: Ruta del modelo
        meses: Lista de meses a evaluar (formato 'YYYY-MM')
        n_muestras: Número de muestras a usar por mes
    Returns:
        DataFrame con columnas: mes, n_samples, f1_score
    """
    from .config import DATA_URLS
    resultados = []
    if meses is None:
        meses = list(DATA_URLS.keys())
    modelo = cargar_modelo(modelo_path)
    for mes in meses:
        try:
            df = pd.read_parquet(DATA_URLS[mes])
            df_proc = build_features(df.head(n_muestras))
            X = df_proc[FEATURES]
            y_true = df_proc[TARGET_COL]
            y_pred = predecir(modelo, X)
            f1 = f1_score(y_true, y_pred)
            resultados.append({
                'mes': mes,
                'n_samples': len(df_proc),
                'f1_score': f1
            })
        except Exception as e:
            resultados.append({
                'mes': mes,
                'n_samples': 0,
                'f1_score': None,
                'error': str(e)
            })
    return pd.DataFrame(resultados)


# Ejemplo de uso si ejecutas este script directamente
if __name__ == "__main__":
    import pandas as pd
    
    # Cargar datos de ejemplo
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet')
    
    # Ejecutar pipeline de predicción
    results = pipeline_prediccion_completo(df)
    
    print("Resultados de predicción:")
    print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"Número de muestras: {results['n_samples']}")
