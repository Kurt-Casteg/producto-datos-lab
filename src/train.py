from sklearn.ensemble import RandomForestClassifier
import joblib

def entrenar_random_forest(X, y, n_estimators=100, max_depth=10, random_state=42):
    """
    Entrena un modelo RandomForestClassifier.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    modelo.fit(X, y)
    return modelo

def guardar_modelo(modelo, ruta_salida):
    """
    Guarda el modelo entrenado en un archivo usando joblib.
    """
    joblib.dump(modelo, ruta_salida)
    print(f"Modelo guardado en {ruta_salida}")

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

    # Entrena el modelo
    modelo = entrenar_random_forest(df_proc[features], df_proc[target_col])

    # Guarda el modelo
    guardar_modelo(modelo, "random_forest.joblib")
