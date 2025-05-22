import joblib
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- 1. Cargar datos ---
df = pd.read_csv("data/outputs/df_cluster.csv")

# --- 2. Mapear edad como ordinal ---
# --- Normalizar edad ---
df['edad'] = df['edad'].astype(str).str.replace("–", "-")  # guion largo a guion corto
df['edad'] = df['edad'].str.replace(r"\s*-\s*", "-", regex=True)  # quitar espacios alrededor del guion
df['edad'] = df['edad'].str.strip()

# --- Mapeo actualizado ---
edad_map = {
    '18-26 años': 1,
    '27-49 años': 2,
    '50-64 años': 3
}
df['edad_ordinal'] = df['edad'].map(edad_map)

# --- 3. Definir columnas ---
numeric_cols = ['edad_ordinal', 'imc', 'totalComidasDia', 'puntaje']
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

# --- 5. Modelos a comparar ---
models = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=4, random_state=42),
    'DBSCAN': DBSCAN(eps=2, min_samples=5),
    'GaussianMixture': GaussianMixture(n_components=4, random_state=42)
}

results = []
labels_dict = {}

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- 6. Entrenar y evaluar todos los modelos ---
for name, model in models.items():
    print(f"\n🔍 Evaluando modelo: {name}")

    try:
        if name == "GaussianMixture":
            clusters = model.fit_predict(X)
        else:
            model.fit(X)
            clusters = model.labels_

        valid_mask = clusters != -1
        if np.sum(valid_mask) == 0:
            print(f"⚠️ {name} no generó clusters válidos. Saltando métricas.")
            continue

        sil_score = silhouette_score(X[valid_mask], clusters[valid_mask])
        ch_score = calinski_harabasz_score(X[valid_mask], clusters[valid_mask])
        db_score = davies_bouldin_score(X[valid_mask], clusters[valid_mask])

        results.append({
            'Modelo': name,
            'Silhouette': sil_score,
            'Calinski-Harabasz': ch_score,
            'Davies-Bouldin': db_score
        })

        # Guardar etiquetas en archivo CSV independiente
        df_copy = df.copy()
        df_copy['cluster'] = clusters
        df_copy.to_csv(f"data/outputs/df_cluster_{name}.csv", index=False)
        print(f"📁 Etiquetas guardadas en: data/outputs/df_cluster_{name}.csv")

        # Guardar modelo .pkl
        joblib.dump(model, f"models/{name}_model.pkl")
        print(f"📦 Modelo guardado en: models/{name}_model.pkl")

    except Exception as e:
        print(f"❌ Error evaluando {name}: {e}")

# --- 7. Guardar tabla de métricas ---
results_df = pd.DataFrame(results)
results_df.to_csv("data/outputs/clustering_comparison.csv", index=False)
print("\n✅ clustering_comparison.csv guardado con las métricas.")

# --- 8. Guardar preprocesador ---
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("📦 Preprocesador guardado como models/preprocessor.pkl")
