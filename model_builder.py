import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- 1. Cargar datos ---
df = pd.read_parquet("data/outputs/df_cluster.parquet")

# --- 2. Mapear edad ---
df['edad'] = df['edad'].astype(str).str.replace("–", "-").str.replace(r"\s*-\s*", "-", regex=True).str.strip()
edad_map = {'18-26 años': 1, '27-49 años': 2, '50-64 años': 3}
df['edad_ordinal'] = df['edad'].map(edad_map)

# Validación opcional
if df['edad_ordinal'].isna().sum() > 0:
    print("Algunas edades no fueron mapeadas correctamente.")

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

# --- 5. Modelos incluyendo DBSCAN adaptativo ---
models = {
    'KMeans': KMeans(n_clusters=4, random_state=42),
    'MiniBatchKMeans': MiniBatchKMeans(n_clusters=4, random_state=42),
    'GaussianMixture': GaussianMixture(n_components=4, random_state=42),
    'DBSCAN': None  # Se ajustará dinámicamente
}

# Función para DBSCAN adaptativo
def get_valid_dbscan(X, eps_values=[0.5, 1.0, 1.5, 2.0, 3.0], min_samples=5):
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        valid_clusters = labels[labels != -1]
        n_clusters = len(np.unique(valid_clusters))
        if n_clusters >= 2:
            print(f" DBSCAN válido con eps={eps} → {n_clusters} clusters.")
            return model, labels
    print(" DBSCAN no encontró clusters válidos.")
    return None, None

# --- 6. Entrenar y evaluar ---
results = []
os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

for name in models:
    print(f"\n Evaluando modelo: {name}")

    try:
        if name == "GaussianMixture":
            model = models[name]
            clusters = model.fit_predict(X)

        elif name == "DBSCAN":
            model, clusters = get_valid_dbscan(X)
            if model is None:
                results.append({
                    'Modelo': name,
                    'Silhouette': None,
                    'Calinski-Harabasz': None,
                    'Davies-Bouldin': None,
                    'n_clusters': 0
                })
                continue
        else:
            model = models[name]
            model.fit(X)
            clusters = model.labels_

        valid_mask = clusters != -1
        valid_clusters = clusters[valid_mask]
        n_clusters = len(np.unique(valid_clusters)) if np.any(valid_mask) else 0

        if n_clusters < 2:
            print(f"{name} generó solo {n_clusters} cluster(s).")
            results.append({
                'Modelo': name,
                'Silhouette': None,
                'Calinski-Harabasz': None,
                'Davies-Bouldin': None,
                'n_clusters': n_clusters
            })
            continue

        # Métricas
        try:
            sil_score = silhouette_score(X[valid_mask], clusters[valid_mask])
            ch_score = calinski_harabasz_score(X[valid_mask], clusters[valid_mask])
            db_score = davies_bouldin_score(X[valid_mask], clusters[valid_mask])
        except Exception as e:
            print(f"No se pudieron calcular métricas para {name}: {e}")
            sil_score, ch_score, db_score = None, None, None

        results.append({
            'Modelo': name,
            'Silhouette': sil_score,
            'Calinski-Harabasz': ch_score,
            'Davies-Bouldin': db_score,
            'n_clusters': n_clusters
        })

        # Guardar etiquetas
        df_copy = df.copy()
        df_copy['cluster'] = clusters
        df_copy.to_csv(f"data/outputs/df_cluster_{name}.csv", index=False)
        print(f"Etiquetas guardadas en: data/outputs/df_cluster_{name}.csv")

        # Guardar modelo
        if model is not None:
            joblib.dump(model, f"models/{name}_model.pkl")
            print(f"Modelo guardado en: models/{name}_model.pkl")

    except Exception as e:
        print(f"Error evaluando {name}: {e}")
        results.append({
            'Modelo': name,
            'Silhouette': None,
            'Calinski-Harabasz': None,
            'Davies-Bouldin': None,
            'n_clusters': 0
        })

# --- 7. Guardar resultados ---
results_df = pd.DataFrame(results)
results_df.to_csv("data/outputs/clustering_comparison.csv", index=False)
print("clustering_comparison.csv guardado.")

# --- 8. Gráfico comparativo ---
results_df.dropna().sort_values('Silhouette', ascending=False).plot(
    x='Modelo', y='Silhouette', kind='bar', legend=False)
plt.title("Comparación de Silhouette Score por Modelo")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("data/outputs/silhouette_comparison.png")
print("Gráfico guardado como silhouette_comparison.png")

# --- 9. Guardar preprocesador ---
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Preprocesador guardado en models/preprocessor.pkl")
