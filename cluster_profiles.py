import pandas as pd
import numpy as np
import os

# --- 1. Cargar archivo del mejor modelo ---
df = pd.read_csv("data/outputs/df_cluster_KMeans.csv")

# --- 2. Limpieza y asegurarse de los tipos ---
numeric_cols = ['edad_ordinal', 'imc', 'totalComidasDia', 'puntaje']
categorical_cols = ['sexo', 'estrato', 'nivel_educativo', 'inseguridad']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[categorical_cols] = df[categorical_cols].astype(str)

# --- 3. Crear resumen por cluster ---
cluster_profiles = df.groupby('cluster').agg({
    'edad_ordinal': ['mean', 'median'],
    'imc': ['mean'],
    'totalComidasDia': ['mean'],
    'puntaje': ['mean'],
    'sexo': lambda x: (x == 'Mujeres').mean() * 100,
    'estrato': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'inseguridad': lambda x: (x.str.lower().str.contains("sí|si")).mean() * 100,
    'cluster': 'count'
})

# Renombrar columnas
cluster_profiles.columns = [
    'Edad promedio', 'Edad mediana', 'IMC promedio', 'Comidas por día',
    'Puntaje IA promedio', '% Mujeres', 'Estrato más común',
    '% con inseguridad alimentaria', 'Tamaño'
]

# Agregar % del total
cluster_profiles['% del total'] = (cluster_profiles['Tamaño'] / cluster_profiles['Tamaño'].sum()) * 100

# Redondear
cluster_profiles = cluster_profiles.round(2)

# --- 4. Exportar resultados ---
os.makedirs("data/outputs", exist_ok=True)
cluster_profiles.to_csv("data/outputs/cluster_profiles_KMeans.csv")
print("✅ Perfiles de clusters generados y guardados en cluster_profiles_KMeans.csv")
