import pandas as pd
import os

# --- 1. Cargar los archivos .dta ---
df_pts = pd.read_stata("data/datasets/PTS_2.dta")
df_ant = pd.read_stata("data/datasets/ANTROPOMETRIA.dta")
df_pisnsp = pd.read_stata("data/datasets/PISNSP.dta")
df_sa = pd.read_stata("data/datasets/SA_1.dta")

# --- 2. Asegurar que las llaves sean texto limpio ---
for df in [df_pts, df_ant, df_pisnsp, df_sa]:
    if 'LLAVE_PERSONA' in df.columns:
        df['LLAVE_PERSONA'] = df['LLAVE_PERSONA'].astype(str).str.strip()
    if 'LLAVE_HOGAR' in df.columns:
        df['LLAVE_HOGAR'] = df['LLAVE_HOGAR'].astype(str).str.strip()

# --- 3. Seleccionar y renombrar columnas ---
df_pts = df_pts[['LLAVE_HOGAR', 'LLAVE_PERSONA', 'edades', 'sexo', 'niveledu', 'cuartil_riqueza2015']].copy()
df_pts.rename(columns={
    'edades': 'edad',
    'niveledu': 'nivel_educativo',
    'cuartil_riqueza2015': 'estrato'
}, inplace=True)

df_ant = df_ant[['LLAVE_PERSONA', 'AN_IMC', 'estadoImc1']].copy()
df_ant.rename(columns={
    'AN_IMC': 'imc',
    'estadoImc1': 'estado_imc'
}, inplace=True)

df_pisnsp = df_pisnsp[['LLAVE_PERSONA', 'totalComidasDia']].copy()
df_sa = df_sa[['LLAVE_HOGAR', 'SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad', 'puntaje']].copy()

# --- 4. Unir todas las fuentes ---
df_cluster = df_pts.merge(df_ant, on='LLAVE_PERSONA', how='inner')
df_cluster = df_cluster.merge(df_pisnsp, on='LLAVE_PERSONA', how='left')
df_cluster = df_cluster.merge(df_sa, on='LLAVE_HOGAR', how='left')
df_cluster.drop(columns=['LLAVE_HOGAR'], inplace=True)

# --- 5. Limpieza de columnas binarias tipo Sí/No ---
ia_cols = ['SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad']
for col in ia_cols:
    df_cluster[col] = (
        df_cluster[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({'sí': 1, 'si': 1, 'no': 0, 'nan': 0, '': 0})
        .fillna(0)
    )
    df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce').fillna(0).astype(int)

# --- 6. Procesar puntaje IA ---
# Asegurar que 'puntaje' es numérico ANTES de renombrar
df_sa['puntaje'] = pd.to_numeric(df_sa['puntaje'], errors='coerce')

# Hacer el merge y luego renombrar con confianza
df_cluster.rename(columns={'puntaje': 'puntaje_ia'}, inplace=True)

# Validar rangos
df_cluster['puntaje_ia'] = df_cluster['puntaje_ia'].fillna(0)
df_cluster.loc[(df_cluster['puntaje_ia'] < 0) | (df_cluster['puntaje_ia'] > 15), 'puntaje_ia'] = 0


# --- 7. Clasificación de inseguridad alimentaria según ENSIN ---
def clasificar_inseguridad(puntaje):
    if puntaje <= 0:
        return 'Seguridad'
    elif 1 <= puntaje <= 4:
        return 'Leve'
    elif 5 <= puntaje <= 8:
        return 'Moderada'
    else:
        return 'Severa'

df_cluster['inseguridad_nivel'] = df_cluster['puntaje_ia'].apply(clasificar_inseguridad)
df_cluster['inseguridad'] = df_cluster['inseguridad_nivel'].replace({
    'Seguridad': 0, 'Leve': 1, 'Moderada': 1, 'Severa': 1
}).astype(int)

# --- 8. Validar tipo de totalComidasDia ---
df_cluster['totalComidasDia'] = pd.to_numeric(df_cluster['totalComidasDia'], errors='coerce').fillna(0)

# --- 9. Eliminar filas con datos esenciales faltantes ---
df_cluster = df_cluster.dropna(subset=['edad', 'sexo', 'nivel_educativo', 'estrato', 'imc', 'estado_imc'])

# --- 10. Exportar datos ---
os.makedirs("data/outputs", exist_ok=True)

# Exportar como CSV con precisión alta
df_cluster.to_csv("data/outputs/df_cluster.csv", index=False, float_format="%.10f")
print("✅ CSV exportado con precisión de 10 decimales.")

# Exportar como Parquet para uso seguro en análisis/modelos
df_cluster.to_parquet("data/outputs/df_cluster.parquet", index=False)
print("✅ Parquet exportado para análisis de machine learning.")

# Información final
print(f"📊 df_cluster tiene {df_cluster.shape[0]} filas y {df_cluster.shape[1]} columnas.")

print(df_cluster['puntaje_ia'].describe())
print(df_cluster['puntaje_ia'].unique()[:20])
