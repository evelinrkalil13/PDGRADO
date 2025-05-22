import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- 1. Cargar métricas y preprocesador ---
results_df = pd.read_parquet("data/outputs/clustering_comparison.parquet")
preprocessor = joblib.load("models/preprocessor.pkl")

# --- 2. Definir columnas numéricas ---
numeric_cols = ['edad_ordinal', 'imc', 'totalComidasDia', 'puntaje_ia']

# --- 3. Analizar todos los modelos ---
for modelo in results_df['Modelo']:
    print(f"\n🔍 Analizando modelo: {modelo}")
    path_parquet = f"data/outputs/df_cluster_{modelo}.parquet"

    if not os.path.exists(path_parquet):
        print(f"❌ Archivo no encontrado: {path_parquet}")
        continue

    df = pd.read_parquet(path_parquet)
    df[numeric_cols + ['cluster']] = df[numeric_cols + ['cluster']].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols + ['cluster']] = df[numeric_cols + ['cluster']].fillna(df[numeric_cols + ['cluster']].mean())

    # --- A. Elbow (solo para KMeans y MiniBatchKMeans) ---
    if modelo in ["KMeans", "MiniBatchKMeans"]:
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(df[numeric_cols])
        X_scaled = StandardScaler().fit_transform(X_imputed)

        sse = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            sse.append(km.inertia_)

        plt.figure(figsize=(6,4))
        plt.plot(range(2,11), sse, marker='o')
        plt.title(f"Elbow Method — {modelo}")
        plt.xlabel("Número de Clusters")
        plt.ylabel("SSE")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"data/outputs/elbow_method_{modelo}.png")
        plt.close()

    # --- B. Silhouette Plot ---
    try:
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(df[numeric_cols])
        X_scaled = StandardScaler().fit_transform(X_imputed)

        sil_vals = silhouette_samples(X_scaled, df['cluster'])
        df['silhouette'] = sil_vals

        plt.figure(figsize=(8,5))
        y_lower = 10
        k = df['cluster'].nunique()
        for i in range(k):
            ith = sil_vals[df['cluster'] == i]
            ith.sort()
            size = len(ith)
            y_upper = y_lower + size
            color = plt.cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, facecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size, str(i))
            y_lower = y_upper + 10
        plt.axvline(np.mean(sil_vals), color="red", linestyle="--")
        plt.xlabel("Coef. Silhouette")
        plt.ylabel("Cluster")
        plt.title(f"Silhouette — {modelo}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"data/outputs/silhouette_plot_{modelo}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ No se pudo graficar Silhouette para {modelo}: {e}")

    # --- C. PCA 2D y 3D ---
    try:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # 2D
        plt.figure(figsize=(6,5))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='tab10', alpha=0.6)
        plt.title(f"PCA 2D — {modelo}")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig(f"data/outputs/pca_2d_{modelo}.png")
        plt.close()

        # 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=df['cluster'], cmap='tab10', alpha=0.6)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        plt.title(f"PCA 3D — {modelo}")
        plt.tight_layout()
        plt.savefig(f"data/outputs/pca_3d_{modelo}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Error en PCA para {modelo}: {e}")

    # --- D. Heatmap de correlación ---
    try:
        corr = df[numeric_cols + ['cluster']].corr()
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Mapa de Correlaciones — {modelo}")
        plt.tight_layout()
        plt.savefig(f"data/outputs/heatmap_correlaciones_{modelo}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ No se pudo graficar heatmap para {modelo}: {e}")

# --- 4. Selección automática del mejor modelo ---
filtered = results_df[(results_df['n_clusters'] >= 2) & (results_df['n_clusters'] <= 10)].dropna()

if filtered.empty:
    print("\n❌ No se encontraron modelos con 2–10 clusters válidos.")
else:
    ranked = filtered.sort_values(by="Silhouette", ascending=False)
    print("\n🏆 Ranking de modelos (clusters entre 2–10):")
    print(ranked[['Modelo', 'Silhouette', 'n_clusters']].to_string(index=False))

    best_model = ranked.iloc[0]
    print(f"\n⭐ Modelo recomendado: {best_model['Modelo']} "
          f"(Silhouette={best_model['Silhouette']:.4f}, "
          f"Clusters={int(best_model['n_clusters'])})")

print("\n✅ Análisis gráfico y selección completados.")
