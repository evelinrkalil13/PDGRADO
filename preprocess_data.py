import pandas as pd
import numpy as np
import os

# --- 1. Cargar archivos .dta ---
df_pts = pd.read_stata("data/datasets/PTS_2.dta")
df_ant = pd.read_stata("data/datasets/ANTROPOMETRIA.dta")
df_pisnsp = pd.read_stata("data/datasets/PISNSP.dta")
df_sa = pd.read_stata("data/datasets/SA_1.dta")

# --- 2. Seleccionar y renombrar columnas ---
df_pts = df_pts[['LLAVE_HOGAR', 'LLAVE_PERSONA', 'edades', 'sexo', 'niveledu', 'cuartil_riqueza2015']].rename(columns={
    'edades': 'edad',
    'niveledu': 'nivel_educativo',
    'cuartil_riqueza2015': 'estrato'
})
df_ant = df_ant[['LLAVE_PERSONA', 'AN_IMC', 'estadoImc1']].rename(columns={
    'AN_IMC': 'imc',
    'estadoImc1': 'estado_imc'
})
df_pisnsp = df_pisnsp[['LLAVE_PERSONA', 'totalComidasDia']]
df_sa = df_sa[['LLAVE_HOGAR', 'SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad', 'puntaje', 'inseguridad']]

# --- 3. Asegurar llaves como texto ---
for df in [df_pts, df_ant, df_pisnsp, df_sa]:
    if 'LLAVE_PERSONA' in df.columns:
        df['LLAVE_PERSONA'] = df['LLAVE_PERSONA'].astype(str).str.strip()
    if 'LLAVE_HOGAR' in df.columns:
        df['LLAVE_HOGAR'] = df['LLAVE_HOGAR'].astype(str).str.strip()

# --- 4. Unir datasets ---
df_cluster = df_pts.merge(df_ant, on='LLAVE_PERSONA', how='inner')
df_cluster = df_cluster.merge(df_pisnsp, on='LLAVE_PERSONA', how='left')
df_cluster = df_cluster.merge(df_sa, on='LLAVE_HOGAR', how='left')
df_cluster.drop(columns=['LLAVE_HOGAR'], inplace=True)

# --- 5. Limpiar columnas binarias ---
bin_cols = ['SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad', 'inseguridad']
for col in bin_cols:
    df_cluster[col] = (
        df_cluster[col].astype(str)
        .str.strip().str.lower()
        .replace({'sí': 1, 'si': 1, 'no': 0, 'nan': 0, '': 0})
        .fillna(0)
    )
    df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce').fillna(0).astype(int)

# --- 6. Procesar puntaje IA ---
df_cluster.rename(columns={'puntaje': 'puntaje_ia'}, inplace=True)
df_cluster['puntaje_ia'] = pd.to_numeric(df_cluster['puntaje_ia'], errors='coerce').fillna(0)

# --- 7. Limpiar totalComidasDia ---
df_cluster['totalComidasDia'] = pd.to_numeric(df_cluster['totalComidasDia'], errors='coerce')
df_cluster.loc[(df_cluster['totalComidasDia'] <= 0.3) | (df_cluster['totalComidasDia'] > 8), 'totalComidasDia'] = np.nan
df_cluster['totalComidasDia'].fillna(df_cluster['totalComidasDia'].mean(), inplace=True)

# --- 8. Eliminar filas con valores esenciales faltantes ---
df_cluster = df_cluster.dropna(subset=['edad', 'sexo', 'nivel_educativo', 'estrato', 'imc', 'estado_imc'])

# --- 9. Crear carpeta y exportar en formato binario ---
os.makedirs("data/outputs", exist_ok=True)
df_cluster.to_parquet("data/outputs/df_cluster.parquet", index=False)
df_cluster.to_pickle("data/outputs/df_cluster.pkl")

# --- 10. Diagnóstico de totalComidasDia ---
print("\n totalComidasDia: valores mínimos y máximos")
print(df_cluster['totalComidasDia'].sort_values().unique()[:10])
print(df_cluster['totalComidasDia'].sort_values(ascending=False).unique()[:10])

print(f"\n df_cluster guardado como .parquet y .pkl con {df_cluster.shape[0]} filas y {df_cluster.shape[1]} columnas.")

# --- Vista previa para validación rápida ---
print("\n Vista previa del dataframe df_cluster:\n")
print(df_cluster.head(5))

print("\n Resumen del DataFrame:")
print(df_cluster.info())
