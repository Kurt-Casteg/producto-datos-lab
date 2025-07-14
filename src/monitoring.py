"""
Módulo para monitoreo y evaluación temporal del modelo.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from .config import DATA_URLS, FEATURES, TARGET_COL
from .features import build_features, detect_concept_drift
from .predict import cargar_modelo, predecir, evaluar_completo

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Clase para monitorear el rendimiento del modelo a lo largo del tiempo."""
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el monitor.
        
        Args:
            model_path: Ruta del modelo
        """
        self.model_path = model_path
        self.model = None
        self.performance_history = {}
        self.drift_history = {}
    
    def load_model(self):
        """Carga el modelo si no está cargado."""
        if self.model is None:
            self.model = cargar_modelo(self.model_path)
    
    def evaluate_month(
        self, 
        month: str, 
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del modelo en un mes específico.
        
        Args:
            month: Mes a evaluar (formato: '2020-01')
            sample_size: Tamaño de muestra para evaluación
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        try:
            # Cargar modelo
            self.load_model()
            
            # Cargar datos
            df = pd.read_parquet(DATA_URLS[month])
            df_proc = build_features(df.head(sample_size))
            
            # Realizar predicciones
            X = df_proc[FEATURES]
            y_true = df_proc[TARGET_COL]
            y_pred = predecir(self.model, X)
            
            # Evaluar
            metrics = evaluar_completo(y_true, y_pred)
            
            # Agregar información adicional
            metrics['month'] = month
            metrics['sample_size'] = len(df_proc)
            metrics['evaluation_date'] = datetime.now().isoformat()
            
            # Guardar en historial
            self.performance_history[month] = metrics
            
            logger.info(f"{month}: F1-Score = {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando {month}: {e}")
            return {'error': str(e), 'month': month}
    
    def evaluate_multiple_months(
        self, 
        months: List[str] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Evalúa el rendimiento en múltiples meses.
        
        Args:
            months: Lista de meses a evaluar
            sample_size: Tamaño de muestra para evaluación
            
        Returns:
            Diccionario con resultados por mes
        """
        if months is None:
            months = list(DATA_URLS.keys())
        
        results = {}
        
        for month in months:
            results[month] = self.evaluate_month(month, sample_size)
        
        return results
    
    def detect_drift(
        self, 
        reference_month: str = '2020-01',
        current_month: str = None
    ) -> Dict[str, Any]:
        """
        Detecta concept drift entre dos meses.
        
        Args:
            reference_month: Mes de referencia
            current_month: Mes actual (si es None, usa el último disponible)
            
        Returns:
            Información sobre concept drift
        """
        if current_month is None:
            available_months = list(DATA_URLS.keys())
            current_month = available_months[-1]
        
        try:
            # Cargar datos de referencia
            df_ref = pd.read_parquet(DATA_URLS[reference_month])
            df_ref_proc = build_features(df_ref.head(1000))
            
            # Cargar datos actuales
            df_curr = pd.read_parquet(DATA_URLS[current_month])
            df_curr_proc = build_features(df_curr.head(1000))
            
            # Detectar drift
            drift_info = detect_concept_drift(df_ref_proc, df_curr_proc)
            
            # Agregar información adicional
            drift_info['reference_month'] = reference_month
            drift_info['current_month'] = current_month
            drift_info['detection_date'] = datetime.now().isoformat()
            
            # Guardar en historial
            self.drift_history[f"{reference_month}_to_{current_month}"] = drift_info
            
            logger.info(f"Drift detectado: {drift_info['drift_ratio']:.2%} de características con drift")
            
            return drift_info
            
        except Exception as e:
            logger.error(f"Error detectando drift: {e}")
            return {'error': str(e)}
    
    def plot_performance_trend(self, save_path: str = None):
        """
        Genera un gráfico de tendencia del rendimiento.
        
        Args:
            save_path: Ruta para guardar el gráfico
        """
        if not self.performance_history:
            logger.warning("No hay datos de rendimiento para graficar")
            return
        
        # Preparar datos
        months = []
        f1_scores = []
        
        for month, metrics in self.performance_history.items():
            if 'error' not in metrics:
                months.append(month)
                f1_scores.append(metrics['f1_score'])
        
        if not months:
            logger.warning("No hay datos válidos para graficar")
            return
        
        # Crear gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(months, f1_scores, marker='o', linewidth=2, markersize=8)
        plt.title('Tendencia del F1-Score por Mes', fontsize=14, fontweight='bold')
        plt.xlabel('Mes', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Agregar línea de umbral
        plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Umbral (0.7)')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()
    
    def plot_drift_analysis(self, save_path: str = None):
        """
        Genera un gráfico de análisis de concept drift.
        
        Args:
            save_path: Ruta para guardar el gráfico
        """
        if not self.drift_history:
            logger.warning("No hay datos de drift para graficar")
            return
        
        # Preparar datos
        drift_ratios = []
        drift_labels = []
        
        for label, drift_info in self.drift_history.items():
            if 'error' not in drift_info:
                drift_ratios.append(drift_info['drift_ratio'])
                drift_labels.append(label)
        
        if not drift_ratios:
            logger.warning("No hay datos válidos de drift para graficar")
            return
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(drift_ratios)), drift_ratios, alpha=0.7)
        plt.title('Análisis de Concept Drift', fontsize=14, fontweight='bold')
        plt.xlabel('Período de Comparación', fontsize=12)
        plt.ylabel('Proporción de Características con Drift', fontsize=12)
        plt.xticks(range(len(drift_labels)), drift_labels, rotation=45)
        
        # Agregar línea de umbral
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral (50%)')
        plt.legend()
        
        # Colorear barras según el nivel de drift
        for i, ratio in enumerate(drift_ratios):
            if ratio > 0.5:
                bars[i].set_color('red')
            elif ratio > 0.3:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo de monitoreo.
        
        Returns:
            Diccionario con reporte completo
        """
        report = {
            'monitoring_date': datetime.now().isoformat(),
            'performance_summary': {},
            'drift_summary': {},
            'recommendations': []
        }
        
        # Resumen de rendimiento
        if self.performance_history:
            f1_scores = [m['f1_score'] for m in self.performance_history.values() if 'error' not in m]
            if f1_scores:
                report['performance_summary'] = {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'min_f1': np.min(f1_scores),
                    'max_f1': np.max(f1_scores),
                    'n_evaluations': len(f1_scores)
                }
        
        # Resumen de drift
        if self.drift_history:
            drift_ratios = [d['drift_ratio'] for d in self.drift_history.values() if 'error' not in d]
            if drift_ratios:
                report['drift_summary'] = {
                    'mean_drift_ratio': np.mean(drift_ratios),
                    'max_drift_ratio': np.max(drift_ratios),
                    'n_drift_analyses': len(drift_ratios)
                }
        
        # Recomendaciones
        if 'performance_summary' in report and report['performance_summary']:
            mean_f1 = report['performance_summary']['mean_f1']
            if mean_f1 < 0.7:
                report['recommendations'].append("El rendimiento promedio está por debajo del umbral recomendado (0.7)")
            
            if 'performance_summary' in report and 'std_f1' in report['performance_summary']:
                std_f1 = report['performance_summary']['std_f1']
                if std_f1 > 0.1:
                    report['recommendations'].append("Alta variabilidad en el rendimiento - considerar retrenamiento")
        
        if 'drift_summary' in report and report['drift_summary']:
            mean_drift = report['drift_summary']['mean_drift_ratio']
            if mean_drift > 0.5:
                report['recommendations'].append("Alto nivel de concept drift detectado - considerar actualización del modelo")
        
        return report


def run_monitoring_pipeline(
    model_path: str = None,
    months: List[str] = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de monitoreo.
    
    Args:
        model_path: Ruta del modelo
        months: Lista de meses a evaluar
        sample_size: Tamaño de muestra para evaluación
        
    Returns:
        Resultados del monitoreo
    """
    logger.info("Iniciando pipeline de monitoreo")
    
    # Crear monitor
    monitor = ModelMonitor(model_path)
    
    # Evaluar rendimiento
    performance_results = monitor.evaluate_multiple_months(months, sample_size)
    
    # Detectar drift
    drift_results = monitor.detect_drift()
    
    # Generar reporte
    report = monitor.generate_report()
    
    # Generar gráficos
    monitor.plot_performance_trend()
    monitor.plot_drift_analysis()
    
    logger.info("Pipeline de monitoreo completado")
    
    return {
        'performance': performance_results,
        'drift': drift_results,
        'report': report
    }


if __name__ == "__main__":
    # Ejecutar pipeline de monitoreo
    results = run_monitoring_pipeline()
    
    print("Reporte de Monitoreo:")
    print(f"Fecha: {results['report']['monitoring_date']}")
    print(f"Evaluaciones realizadas: {results['report']['performance_summary'].get('n_evaluations', 0)}")
    print(f"Análisis de drift: {results['report']['drift_summary'].get('n_drift_analyses', 0)}")
    
    if results['report']['recommendations']:
        print("\nRecomendaciones:")
        for rec in results['report']['recommendations']:
            print(f"- {rec}") 