import joblib
from sklearn.metrics import f1_score

def cargar_modelo(ruta_modelo):
    """
    Carga un modelo serializado con joblib.
    """
    modelo = joblib.load(ruta_modelo)
    print(f"Modelo cargado desde {ruta_modelo}")
    return modelo

def predecir(modelo, X):
    """
    Realiza predicciones de probabilidad y devuelve las etiquetas predichas.
    """
    # predict_proba devuelve probabilidades, tomamos la de la clase positiva
    probas = modelo.predict_proba(X)
    # Redondeamos la probabilidad de la clase positiva para obtener la etiqueta
    etiquetas = [p[1] for p in probas.round()]
    return etiquetas

def evaluar(y_true, y_pred):
    """
    Calcula el F1-score.
    """
    f1 = f1_score(y_true, y_pred)
    print(f"F1-score: {f1}")
    return f1

# Ejemplo de uso si ejecutas este script directamente
if __name__ == "__main__":
    import pandas as pd
    from src.dataset import preprocess  # Asegúrate de tener esta función en dataset.py

    # Carga tus datos aquí (ajusta la ruta y el nombre del archivo según tu caso)
    df = pd.read_parquet('ruta/a/tu/archivo.parquet')
    features = [...]      # Lista de nombres de columnas de características
    target_col = "high_tip"  # Nombre de la columna objetivo

    # Preprocesa los datos
    df_proc = preprocess(df, target_col, features)

    # Carga el modelo
    modelo = cargar_modelo("random_forest.joblib")

    # Predice
    y_pred = predecir(modelo, df_proc[features])

    # Evalúa
    evaluar(df_proc[target_col], y_pred)
