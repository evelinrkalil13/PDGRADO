
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
├── main.py
├── recommender.py
├── preprocess_data.py
├── model_builder.py
├── model_analisis.py
├── cluster_profiles.py
│
├── models/
│   ├── KMeans_model.pkl
│   ├── preprocessor.pkl
│   ├── DBSCAN_model.pkl
│   ├── GaussianMixture_model.pkl
│   └── MiniBatchKMeans_model.pkl
│
├── data/
│   └── outputs/
│       ├── df_cluster_KMeans.parquet
│       ├── df_cluster_DBSCAN.parquet
│       ├── df_cluster_GaussianMixture.parquet
│       ├── df_cluster_MiniBatchKMeans.parquet
│       ├── df_cluster.parquet
│       ├── df_cluster.pkl
│       └── clustering_comparison.parquet
│
├── template/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── images/
│       ├── background.jpg
│       └── icono.ico
│
└── requirements.txt
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

### Endpoint disponible
```
POST http://localhost:8000/predecir
```

### JSON de entrada

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

## Informe Final del Proyecto

### Portada
**Título:** Perfiles de Inseguridad Alimentaria para Recomendaciones Nutricionales 
**Curso:** Optativa II 
**Fecha:** 2025  
**Integrantes:** Evelyn Rendón Kalil

### Introducción
El proyecto aborda el problema de clasificar usuarios según características alimentarias y sociodemográficas, identificando patrones de riesgo y proponiendo recomendaciones nutricionales automatizadas.  
**Objetivo:** Desarrollar una herramienta de ayuda basada en clustering.  
**Resumen:** Aplicación FastAPI con modelo ML, interfaz visual y endpoint funcional.

### Desarrollo del Modelo ML
- **Datos:** Provienen de encuesta nutricional ENSIN y respuestas tipo ELCSA.
- **Preprocesamiento:** Limpieza, codificación, escalamiento.
- **Entrenamiento:** KMeans,MiniBatchKMeans, DBSCAN, GaussianMixture.
- **Evaluación:** Elbow method, Silhouette Score, Heatmap, PCA, análisis visual.
- **Resultado final:** Se seleccionó KMeans por rendimiento y segmentación clara.

### Desarrollo de la API (FastAPI)
- Endpoint `/clasificar_usuario`: procesa formulario.
- Endpoint `/predecir`: consume JSON y devuelve cluster + recomendación.
- Uso de Joblib para cargar modelos y transformadores.

### Desarrollo de la GUI
- HTML con Jinja2 y Bootstrap.
- Glassmorphism moderno y responsivo.
- Validación integrada.
- Popup con resultados.

### Arquitectura del Sistema
FastAPI ↔ Modelo ML + Preprocesamiento  
        ↕  
HTML Formulario ↔ Recomendación

### Discusión
- **Desafíos:** limpieza de datos, validación, adaptación de formularios.
- **Limitaciones:** modelo no adaptativo en tiempo real.
- **Ética:** tratamiento responsable de datos sensibles.
- **Mejoras:** sistema de autenticación, base de datos.

---

## Créditos

Este proyecto fue desarrollado como parte de un trabajo académico para el curso de OPTATIVA II.
