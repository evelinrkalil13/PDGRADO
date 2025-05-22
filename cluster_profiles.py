import pandas as pd

# --- 1. Cargar el archivo ---
df = pd.read_csv("data/outputs/df_cluster_KMeans.csv")

# --- 2. Corrección de codificación ---
categorical_cols = ['edad', 'sexo', 'nivel_educativo', 'estrato', 'estado_imc', 'inseguridad']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.encode('utf-8').str.decode('utf-8', errors='ignore').str.lower()

# --- 3. Seleccionar un representante por cluster ---
ejemplares = df.groupby("cluster").apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

# --- 4. Mostrar salida narrativa ---
print("\n🧍‍♂️ Narrativas representativas por cluster\n" + "-"*60)

for i, row in ejemplares.iterrows():
    cluster = int(row['cluster'])
    edad = row['edad']
    sexo = row['sexo']
    educ = row['nivel_educativo']
    estrato = row['estrato']
    imc = row['imc']
    imc_estado = row['estado_imc']
    comidas = row['totalComidasDia']
    puntaje = row['puntaje_ia']
    inseguridad = "sí" if int(row['inseguridad']) == 1 else "no"

    # --- Nivel de inseguridad basado en puntaje IA ---
    if puntaje >= 10:
        inseg_text = "alta"
    elif puntaje >= 5:
        inseg_text = "moderada"
    elif puntaje > 0:
        inseg_text = "leve"
    else:
        inseg_text = "nula"

    print(f"\n🔹 Cluster {cluster}")
    print(f"Este grupo está representado por una persona de sexo {sexo}, entre {edad}, con nivel educativo '{educ}' y perteneciente al estrato '{estrato}'.")
    print(f"Presenta un IMC de {imc:.2f} ({imc_estado}), consume aproximadamente {comidas} comidas al día y tiene un puntaje IA de {puntaje} (inseguridad {inseg_text}).")
    print(f"Respuestas afirmativas en preguntas tipo ELCSA: {inseguridad.capitalize()}.")

print("\n✅ Narrativas completadas.")
