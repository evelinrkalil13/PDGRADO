import pandas as pd

# --- 1. Cargar archivo de clusters ---
df = pd.read_csv("data/outputs/df_cluster_KMeans.csv")

# --- 2. Corregir codificación y tipos ---
categorical_cols = ['edad', 'sexo', 'nivel_educativo', 'estrato', 'estado_imc', 'inseguridad']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.encode('utf-8').str.decode('utf-8', errors='ignore').str.lower()

# --- 3. Seleccionar un usuario representativo por cluster ---
ejemplares = df.groupby("cluster").apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

# --- 4. Mostrar resultados directamente por consola ---
print("\n Usuarios representativos por cluster:\n" + "-"*50)

for i, row in ejemplares.iterrows():
    print(f"\n🔹 Cluster {int(row['cluster'])}")
    print(f"• Edad: {row['edad']} (ordinal = {row['edad_ordinal']})")
    print(f"• Sexo: {row['sexo']}")
    print(f"• Nivel educativo: {row['nivel_educativo']}")
    print(f"• Estrato socioeconómico: {row['estrato']}")
    print(f"• IMC: {row['imc']} — Estado IMC: {row['estado_imc']}")
    print(f"• Total comidas por día: {row['totalComidasDia']}")
    print(f"• Puntaje IA: {row['puntaje_ia']}")
    print(f"• Inseguridad alimentaria (preguntas tipo sí/no): {'Sí' if int(row['inseguridad']) == 1 else 'No'}")


