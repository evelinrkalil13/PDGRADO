import pandas as pd
import os

# --- 1. Cargar los archivos .dta ---
df_pts = pd.read_stata("data/PTS_2.dta")
df_ant = pd.read_stata("data/ANTROPOMETRIA.dta")
df_pisnsp = pd.read_stata("data/PISNSP.dta")
df_sa = pd.read_stata("data/SA_1.dta")

# --- 2. Seleccionar columnas relevantes ---
df_pts = df_pts[['LLAVE_HOGAR', 'LLAVE_PERSONA', 'edades', 'sexo', 'niveledu', 'cuartil_riqueza2015']]
df_pts.rename(columns={
    'edades': 'edad',
    'niveledu': 'nivel_educativo',
    'cuartil_riqueza2015': 'estrato'  
}, inplace=True)

df_ant = df_ant[['LLAVE_PERSONA', 'AN_IMC', 'estadoImc1']]
df_ant.rename(columns={
    'AN_IMC': 'imc',
    'estadoImc1': 'estado_imc'
}, inplace=True)

df_pisnsp = df_pisnsp[['LLAVE_PERSONA', 'totalComidasDia']]

df_sa = df_sa[['LLAVE_HOGAR', 'SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad', 'puntaje', 'inseguridad']]

# --- 3. Unir todo en df_cluster ---
df_cluster = df_pts.merge(df_ant, on='LLAVE_PERSONA', how='inner')
df_cluster = df_cluster.merge(df_pisnsp, on='LLAVE_PERSONA', how='left')
df_cluster = df_cluster.merge(df_sa, on='LLAVE_HOGAR', how='left')
df_cluster.drop(columns=['LLAVE_HOGAR'], inplace=True)

# --- 4. Limpiar y convertir columnas binarizadas tipo "Sí/No" ---
ia_cols = ['SA10_1', 'SA11_1', 'disminuir_porciones', 'pedir_prestado', 'menor_calidad', 'inseguridad']
for col in ia_cols:
    df_cluster[col] = (
        df_cluster[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({'sí': 1, 'si': 1, 'no': 0, 'nan': 0, '': 0})
        .fillna(0))

# Convertir solo si todo es numérico
df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce').fillna(0).astype(int)


# --- 5. Procesar puntaje IA ---
df_cluster.rename(columns={'puntaje': 'puntaje_ia'}, inplace=True)
df_cluster['puntaje_ia'] = pd.to_numeric(df_cluster['puntaje_ia'], errors='coerce').fillna(0)

# --- 6. Eliminar filas con datos esenciales faltantes ---
df_cluster = df_cluster.dropna(subset=['edad', 'sexo', 'nivel_educativo', 'estrato', 'imc', 'estado_imc'])

# --- 7. Exportar ---
os.makedirs("data/outputs", exist_ok=True)
df_cluster.to_csv("data/outputs/df_cluster.csv", index=False)
print(f"✅ df_cluster generado en data/outputs/ con {df_cluster.shape[0]} registros y {df_cluster.shape[1]} columnas.")
