"""
Módulo para el procesamiento de características y preprocesamiento de datos.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats
from .config import (
    FEATURES, 
    NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES, 
    TARGET_COL, 
    EPS, 
    TIP_THRESHOLD
)


def build_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Construye características a partir de datos de viajes en taxi.
    
    Args:
        df: DataFrame con datos de viajes
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame procesado con características
    """
    # Limpieza básica
    df = df[df['fare_amount'] > 0].reset_index(drop=True)
    
    # Crear variable objetivo
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    df[target_col] = df['tip_fraction'] > TIP_THRESHOLD

    # Construir características
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (
        (df['pickup_weekday'] >= 0) & 
        (df['pickup_weekday'] <= 4) & 
        (df['pickup_hour'] >= 8) & 
        (df['pickup_hour'] <= 18)
    )
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.seconds
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + EPS)

    # Seleccionar columnas relevantes
    df = df[['tpep_dropoff_datetime'] + FEATURES + [target_col]]
    df[FEATURES + [target_col]] = df[FEATURES + [target_col]].astype("float32").fillna(-1.0)

    # Convertir objetivo a int32 para eficiencia
    df[target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Función de preprocesamiento principal (alias de build_features para compatibilidad).
    
    Args:
        df: DataFrame con datos de viajes
        target_col: Nombre de la columna objetivo
        
    Returns:
        DataFrame procesado con características
    """
    return build_features(df, target_col)


def compare_distributions(dfa: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
    """
    Compara las distribuciones de dos datasets usando la prueba de Kolmogorov-Smirnov.
    
    Args:
        dfa: Primer DataFrame
        dfb: Segundo DataFrame
        
    Returns:
        DataFrame con estadísticas de comparación
    """
    statistics = []
    p_values = []

    for feature in FEATURES:
        statistic, p_value = stats.ks_2samp(dfa[feature], dfb[feature])
        statistics.append(statistic)
        p_values.append(p_value)
    
    comparison_df = pd.DataFrame({
        'feature': FEATURES, 
        'statistic': statistics, 
        'p_value': p_values
    })
    
    return comparison_df.sort_values(by='p_value', ascending=True)


def detect_concept_drift(
    reference_df: pd.DataFrame, 
    current_df: pd.DataFrame, 
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Detecta concept drift comparando distribuciones de características.
    
    Args:
        reference_df: DataFrame de referencia
        current_df: DataFrame actual
        alpha: Nivel de significancia
        
    Returns:
        Diccionario con información sobre concept drift
    """
    comparison = compare_distributions(reference_df, current_df)
    
    # Contar características con drift significativo
    drifted_features = comparison[comparison['p_value'] < alpha]
    
    drift_info = {
        'total_features': len(FEATURES),
        'drifted_features': len(drifted_features),
        'drift_ratio': len(drifted_features) / len(FEATURES),
        'drifted_feature_names': drifted_features['feature'].tolist(),
        'comparison_results': comparison
    }
    
    return drift_info


def validate_features(features_array: np.ndarray) -> bool:
    """
    Valida que el array de características tenga el formato correcto.
    
    Args:
        features_array: Array de características
        
    Returns:
        True si el formato es válido, False en caso contrario
    """
    if features_array.shape[1] != len(FEATURES):
        return False
    
    # Verificar que no haya valores NaN o infinitos
    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
        return False
    
    return True


def get_feature_names() -> List[str]:
    """
    Retorna la lista de nombres de características.
    
    Returns:
        Lista de nombres de características
    """
    return FEATURES.copy()


def get_numeric_features() -> List[str]:
    """
    Retorna la lista de características numéricas.
    
    Returns:
        Lista de características numéricas
    """
    return NUMERIC_FEATURES.copy()


def get_categorical_features() -> List[str]:
    """
    Retorna la lista de características categóricas.
    
    Returns:
        Lista de características categóricas
    """
    return CATEGORICAL_FEATURES.copy()