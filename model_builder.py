import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- 1. Cargar datos preprocesados desde parquet ---
df = pd.read_parquet("data/outputs/df_cluster.parquet")

# --- 2. Mapear edad ordinal ---
edad_map = {
    '18 – 26 años': 1,
    '27 - 49 años': 2,
    '50 – 64 años': 3
}
df['edad_ordinal'] = df['edad'].map(edad_map)

# --- 3. Definir columnas ---
numeric_cols = ['edad_ordinal', 'imc', 'totalComidasDia', 'puntaje_ia']
categorical_cols = ['sexo', 'nivel_educativo', 'estado_imc', 'estrato', 'inseguridad']

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[categorical_cols] = df[categorical_cols].astype(str)

# --- 4. Preprocesamiento ---
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols)
])

X = preprocessor.fit_transform(df)

# --- 5. Definir modelos ---
models = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=4, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'GaussianMixture': GaussianMixture(n_components=4, random_state=42)
}

results = []
os.makedirs("models", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

# --- 6. Entrenar y evaluar modelos ---
for name, model in models.items():
    print(f"\n Evaluando modelo: {name}")
    try:
        if name == "GaussianMixture":
            clusters = model.fit_predict(X)
        else:
            model.fit(X)
            clusters = model.labels_

        valid_mask = clusters != -1
        if np.sum(valid_mask) < 2:
            print(f" {name} no generó clusters válidos.")
            continue

        sil = silhouette_score(X[valid_mask], clusters[valid_mask])
        ch = calinski_harabasz_score(X[valid_mask], clusters[valid_mask])
        db = davies_bouldin_score(X[valid_mask], clusters[valid_mask])

        results.append({
            'Modelo': name,
            'Silhouette': sil,
            'Calinski-Harabasz': ch,
            'Davies-Bouldin': db,
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
        })

        df_result = df.copy()
        df_result['cluster'] = clusters
        df_result.to_parquet(f"data/outputs/df_cluster_{name}.parquet", index=False)
        joblib.dump(model, f"models/{name}_model.pkl")

        print(f" Etiquetas guardadas en Parquet: df_cluster_{name}.parquet")
        print(f" Modelo guardado en: models/{name}_model.pkl")

    except Exception as e:
        print(f" Error con {name}: {e}")

# --- 7. Guardar métricas comparativas ---
results_df = pd.DataFrame(results)
results_df.to_parquet("data/outputs/clustering_comparison.parquet", index=False)
print("\n Métricas guardadas en clustering_comparison.parquet")

# --- 8. Guardar preprocesador ---
joblib.dump(preprocessor, "models/preprocessor.pkl")
print(" Preprocesador guardado en: models/preprocessor.pkl")
