import pandas as pd
import joblib
import numpy as np

# --- 1. Cargar modelo y preprocesador entrenados ---
model = joblib.load("models/KMeans_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# --- 2. Cargar data original para inferir perfiles por cluster ---
df = pd.read_parquet("data/outputs/df_cluster_KMeans.parquet")

# --- 3. Mapear perfiles y recomendaciones por cluster ---
def construir_perfiles(df):
    # Forzar numéricos con conversión segura
    numeric_cols = ["edad_ordinal", "imc", "totalComidasDia", "puntaje_ia"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # fuerza float o NaN

    # Asegurar que 'inseguridad' sea binario entero
    df['inseguridad'] = pd.to_numeric(df['inseguridad'], errors="coerce").fillna(0).astype(int)

    # Agrupación segura
    resumen = df.groupby("cluster").agg({
        "edad_ordinal": "mean",
        "imc": "mean",
        "totalComidasDia": "mean",
        "puntaje_ia": "mean",
        "sexo": lambda x: (x == "mujeres").mean() * 100,
        "estrato": lambda x: x.mode()[0] if not x.mode().empty else "N/D",
        "inseguridad": "mean",
        "cluster": "count"
    }).rename(columns={"cluster": "tamaño"})

    resumen["% mujeres"] = resumen.pop("sexo").round(1)
    resumen["% inseguridad"] = (resumen.pop("inseguridad") * 100).round(1)
    return resumen


perfiles = construir_perfiles(df)

# --- 4. Reglas básicas por cluster ---
recomendaciones = {
    0: "Promover balance nutricional y control de porciones.",
    1: "Aumentar calidad calórica y monitorear bajo peso.",
    2: "Asistencia en calidad alimentaria y reducir ultraprocesados.",
    3: "Reeducación nutricional en mujeres jóvenes vulnerables."
}

# --- 5. Función principal ---
def recomendar(usuario: dict) -> dict:
    try:
        user_df = pd.DataFrame([usuario])

        # Asegurar columnas necesarias
        required_cols = ['edad_ordinal', 'imc', 'totalComidasDia', 'puntaje_ia',
                         'sexo', 'nivel_educativo', 'estado_imc', 'estrato', 'inseguridad']
        for col in required_cols:
            if col not in user_df.columns:
                raise ValueError(f"Falta la columna: {col}")

        # Transformar
        X_new = preprocessor.transform(user_df)

        # Predecir
        cluster = int(model.predict(X_new)[0])
        perfil = perfiles.loc[cluster].to_dict()
        recomendacion = recomendaciones.get(cluster, "Personalizar recomendaciones.")

        return {
            "cluster": cluster,
            "perfil_resumido": perfil,
            "recomendacion": recomendacion
        }

    except Exception as e:
        return {"error": str(e)}
