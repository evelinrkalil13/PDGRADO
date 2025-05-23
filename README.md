
# Recomendador Nutricional

Este proyecto es una aplicación web desarrollada con **FastAPI**, que permite al usuario ingresar sus características demográficas y alimentarias para obtener:

- El **perfil nutricional** correspondiente (a partir de clustering).
- Una **recomendación personalizada** con base en su grupo de riesgo alimentario.

La aplicación emplea un modelo de Machine Learning previamente entrenado (KMeans), y ofrece tanto una interfaz web amigable como un endpoint para integración vía API.

---

## Funcionalidades

- Clasificación de usuarios según hábitos alimentarios y características sociodemográficas.
- Cálculo automático del IMC y la inseguridad alimentaria con base en preguntas tipo ELCSA.
- Interfaz web moderna con diseño tipo glassmorphism.
- Endpoint JSON accesible desde Thunder Client, Postman o cualquier sistema externo.
- Resultados mostrados con recomendaciones personalizadas.

---

## Tecnologías utilizadas

- Python 3.10+
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Jinja2 (para templates HTML)
- Uvicorn
- HTML + CSS (Glassmorphism + Bootstrap 5)

---

## Estructura del proyecto

```
RecomendadorNutricional/
│
├── main.py                  # Aplicación principal de FastAPI
├── recommender.py           # Lógica de predicción y recomendaciones
├── preprocess_data.py       # Preprocesamiento de los datos
├── model_builder.py         # Constructor de modelos
├── model_analisis.py        # Análisis de los modelos
├── cluster_profiles.py      # Análisis de los clusters
│
├── models/
│   ├── KMeans_model.pkl     # Modelo de clustering entrenado
│   ├── preprocessor.pkl     # Pipeline de preprocesamiento
│   ├── DBSCAN_model.pkl
│   ├── GaussianMixture_model.pkl
│   └── MiniBatchKMeans_model.pkl
│
├── data/
│   └── outputs/
│       ├── df_cluster_KMeans.parquet  # Datos etiquetados para resumen de perfiles
│       ├── df_cluster_KMeans.parquet
│       ├── df_cluster_DBSCAN.parquet
│       ├── df_cluster_GaussianMixture.parquet
│       ├── df_cluster_MiniBatchKMeans.parquet
│       ├── df_cluster.parquet
│       ├── df_cluster.pkl
│       └── clustering_comparison.parquet
│
├── template/
│   └── index.html           # Formulario HTML estilizado
│
├── static/
│   ├── style.css            # Estilos personalizados
│   └── images/
│       ├── background.jpg   # Imagen de fondo
│       └── icono.ico   # Imagen de fondo
│
└── requirements.txt         # Dependencias necesarias
```

---

## Cómo ejecutar el proyecto localmente

1. **Clonar el repositorio**  
```bash
git clone https://github.com/tu_usuario/recomendador-nutricional.git
cd recomendador-nutricional
```

2. **Crear entorno virtual**  
```bash
python -m venv .venv
```

3. **Activar entorno virtual**  
- En Windows:  
```bash
.venv\Scripts\activate
```
- En macOS/Linux:  
```bash
source .venv/bin/activate
```

4. **Instalar dependencias**  
```bash
pip install -r requirements.txt
```

5. **Lanzar aplicación local**  
```bash
uvicorn main:app --reload
```

6. **Acceder desde el navegador**  
[http://localhost:8000](http://localhost:8000)

---

## Uso del endpoint en Thunder Client / Postman

Además de la interfaz visual, la aplicación expone un endpoint para integración con herramientas externas:

### Endpoint disponible
```
POST http://localhost:8000/predecir
```

### JSON de entrada (lista de usuarios)

```json
[
  {
    "edad": "18 - 26 años",
    "sexo": "mujeres",
    "nivel_educativo": "primaria incompleta",
    "estrato": "primer cuartil",
    "peso": 60,
    "altura": 160,
    "totalComidasDia": 3,
    "sa10_1": 1,
    "sa11_1": 0,
    "menor_calidad": 1
  }
]
```

### Respuesta esperada

```json
{
  "message": "Clasificación realizada correctamente.",
  "resultados": [
    {
      "cluster": 2,
      "recomendacion": "Asistencia en calidad alimentaria y reducir ultraprocesados."
    }
  ]
}
```

---

## Consideraciones

- Todos los campos son obligatorios.
- La **altura debe estar en centímetros** (ej. 160).
- Las respuestas tipo ELCSA (`sa10_1`, `sa11_1`, `menor_calidad`) deben ser `0` o `1`.
- La API devolverá un error si falta algún campo o el tipo de dato es incorrecto.

---

## Créditos

Este proyecto fue desarrollado como parte de un trabajo académico sobre hábitos alimentarios y perfilamiento basado en datos, en el marco de estudios en **Analítica y Machine Learning**.
