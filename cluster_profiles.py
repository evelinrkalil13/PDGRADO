import pandas as pd

# --- 1. Cargar archivo parquet ---
df = pd.read_parquet("data/outputs/df_cluster_KMeans.parquet")

# --- 2. Normalizar texto y limpieza general ---
categorical_cols = ['edad', 'sexo', 'nivel_educativo', 'estrato', 'estado_imc', 'inseguridad']
for col in categorical_cols:
    df[col] = (
        df[col].astype(str)
        .str.strip()
        .str.encode('utf-8')
        .str.decode('utf-8', errors='ignore')
        .str.lower()
    )

# --- 3. Mapear edad ordinal ---
df['edad'] = df['edad'].str.replace("–", "-").str.replace(r"\s*-\s*", "-", regex=True).str.strip()
edad_map = {'18-26 años': 1, '27-49 años': 2, '50-64 años': 3}
df['edad_ordinal'] = df['edad'].map(edad_map)

# --- 4. Seleccionar un ejemplo representativo por cluster ---
ejemplares = df.groupby("cluster", group_keys=False).apply(
    lambda x: x.sample(1, random_state=42)
).sort_values("cluster").reset_index(drop=True)

# --- 5. Mostrar resultados por cluster ---
print("\n Resumen representativo de cada cluster\n" + "-"*60)

for _, row in ejemplares.iterrows():
    cluster = int(row['cluster'])
    edad = row['edad']
    edad_ordinal = row['edad_ordinal']
    edad_ordinal_str = f"{int(edad_ordinal)}" if pd.notna(edad_ordinal) else "N/D"
    sexo = row['sexo']
    educ = row['nivel_educativo']
    estrato = row['estrato']
    imc = row['imc']
    estado_imc = row['estado_imc']
    comidas = row['totalComidasDia']
    puntaje = row['puntaje_ia']
    inseguridad_si = "sí" if int(row['inseguridad']) == 1 else "no"

    # Inseguridad descriptiva
    if puntaje >= 10:
        inseg_desc = "alta"
    elif puntaje >= 5:
        inseg_desc = "moderada"
    elif puntaje > 0:
        inseg_desc = "leve"
    else:
        inseg_desc = "nula"

    print(f"\n - Cluster {cluster}")
    print(f"• Edad: {edad} (ordinal = {edad_ordinal_str})")
    print(f"• Sexo: {sexo}")
    print(f"• Nivel educativo: {educ}")
    print(f"• Estrato socioeconómico: {estrato}")
    print(f"• IMC: {imc:.2f} — Estado IMC: {estado_imc}")
    print(f"• Total comidas por día: {comidas:.2f}")
    print(f"• Puntaje IA: {puntaje:.1f} → Inseguridad {inseg_desc}")
    print(f"• Inseguridad alimentaria (preguntas tipo sí/no): {inseguridad_si.capitalize()}")

# --- 6. Resumen final por tamaño ---
print("\n Tamaño de cada cluster:")
print(df['cluster'].value_counts().sort_index())
