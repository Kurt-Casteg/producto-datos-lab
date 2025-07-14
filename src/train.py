"""
Módulo para el entrenamiento de modelos de machine learning.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple, Optional
import logging

from .config import MODEL_CONFIG, MODEL_PATH, FEATURES, TARGET_COL
from .features import build_features

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def entrenar_random_forest(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_estimators: int = MODEL_CONFIG['n_estimators'],
    max_depth: int = MODEL_CONFIG['max_depth'], 
    random_state: int = MODEL_CONFIG['random_state']
) -> RandomForestClassifier:
    """
    Entrena un modelo RandomForestClassifier.
    
    Args:
        X: Características de entrenamiento
        y: Variable objetivo
        n_estimators: Número de árboles
        max_depth: Profundidad máxima de los árboles
        random_state: Semilla para reproducibilidad
        
    Returns:
        Modelo entrenado
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    logger.info(f"Entrenando RandomForest con {n_estimators} árboles y profundidad máxima {max_depth}")
    modelo.fit(X, y)
    
    return modelo


def evaluar_modelo(
    modelo: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evalúa el rendimiento del modelo.
    
    Args:
        modelo: Modelo entrenado
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        Diccionario con métricas de evaluación
    """
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    accuracy = modelo.score(X_test, y_test)
    
    metrics = {
        'f1_score': f1,
        'accuracy': accuracy,
        'n_samples': len(X_test)
    }
    
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return metrics


def guardar_modelo(modelo: RandomForestClassifier, ruta_salida: str = MODEL_PATH) -> None:
    """
    Guarda el modelo entrenado en un archivo usando joblib.
    
    Args:
        modelo: Modelo a guardar
        ruta_salida: Ruta donde guardar el modelo
    """
    joblib.dump(modelo, ruta_salida)
    logger.info(f"Modelo guardado en {ruta_salida}")


def cargar_modelo(ruta_modelo: str = MODEL_PATH) -> RandomForestClassifier:
    """
    Carga un modelo serializado con joblib.
    
    Args:
        ruta_modelo: Ruta del modelo a cargar
        
    Returns:
        Modelo cargado
    """
    modelo = joblib.load(ruta_modelo)
    logger.info(f"Modelo cargado desde {ruta_modelo}")
    return modelo


def entrenar_y_evaluar(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Entrena un modelo y evalúa su rendimiento.
    
    Args:
        df: DataFrame con datos procesados
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla con modelo entrenado y métricas de evaluación
    """
    # Dividir datos
    X = df[FEATURES]
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Entrenar modelo
    modelo = entrenar_random_forest(X_train, y_train)
    
    # Evaluar modelo
    metrics = evaluar_modelo(modelo, X_test, y_test)
    
    return modelo, metrics


def cross_validation_evaluation(
    modelo: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, float]:
    """
    Realiza evaluación con validación cruzada.
    
    Args:
        modelo: Modelo a evaluar
        X: Características
        y: Variable objetivo
        cv: Número de folds
        
    Returns:
        Diccionario con métricas de validación cruzada
    """
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1')
    
    cv_metrics = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'cv_f1_scores': cv_scores.tolist()
    }
    
    logger.info(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_metrics


def pipeline_entrenamiento_completo(
    df_raw: pd.DataFrame,
    guardar: bool = True,
    ruta_salida: str = MODEL_PATH
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Pipeline completo de entrenamiento desde datos crudos.
    
    Args:
        df_raw: DataFrame con datos crudos
        guardar: Si guardar el modelo
        ruta_salida: Ruta donde guardar el modelo
        
    Returns:
        Tupla con modelo entrenado y métricas completas
    """
    logger.info("Iniciando pipeline de entrenamiento completo")
    
    # Preprocesar datos
    df_proc = build_features(df_raw)
    logger.info(f"Datos preprocesados: {len(df_proc)} muestras")
    
    # Entrenar y evaluar
    modelo, metrics = entrenar_y_evaluar(df_proc)
    
    # Validación cruzada
    X = df_proc[FEATURES]
    y = df_proc[TARGET_COL]
    cv_metrics = cross_validation_evaluation(modelo, X, y)
    
    # Combinar métricas
    all_metrics = {**metrics, **cv_metrics}
    
    # Guardar modelo si se solicita
    if guardar:
        guardar_modelo(modelo, ruta_salida)
    
    logger.info("Pipeline de entrenamiento completado")
    
    return modelo, all_metrics


# Ejemplo de uso si ejecutas este script directamente
if __name__ == "__main__":
    import pandas as pd
    
    # Cargar datos de ejemplo
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet')
    
    # Ejecutar pipeline completo
    modelo, metrics = pipeline_entrenamiento_completo(df)
    
    print("Métricas finales:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
