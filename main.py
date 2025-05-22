# main.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from recommender import recomendar

import pandas as pd

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def form_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/clasificar_usuario", response_class=HTMLResponse)
async def clasificar_usuario(
    request: Request,
    edad: str = Form(...),
    sexo: str = Form(...),
    nivel_educativo: str = Form(...),
    estrato: str = Form(...),
    peso: float = Form(...),
    altura: float = Form(...),
    totalComidasDia: float = Form(...),
    sa10_1: str = Form(...),
    sa11_1: str = Form(...),
    menor_calidad: str = Form(...)
):
    # --- Mapear edad a ordinal ---
    edad_map = {
        '18 - 26 años': 1,
        '27 - 49 años': 2,
        '50 - 64 años': 3
    }
    edad_ordinal = edad_map.get(edad.strip(), None)

    # --- Calcular IMC y estado ---
    imc = peso / (altura ** 2)
    if imc < 18.5:
        estado_imc = "delgadez"
    elif imc < 25:
        estado_imc = "normal"
    else:
        estado_imc = "exceso de peso"

    # --- Convertir respuestas tipo ELCSA ---
    #sa10_1_val = 1 if sa10_1.lower() == "sí" else 0
    #sa11_1_val = 1 if sa11_1.lower() == "sí" else 0
    #menor_calidad_val = 1 if menor_calidad.lower() == "sí" else 0

    sa10_1_val = int(sa10_1)
    sa11_1_val = int(sa11_1)
    menor_calidad_val = int(menor_calidad)

    print(sa10_1, sa11_1)

    # --- Calcular puntaje IA e inseguridad ---
    puntaje_ia = sa10_1_val + sa11_1_val + menor_calidad_val
    inseguridad = 1 if puntaje_ia > 0 else 0

    usuario = {
        "edad_ordinal": edad_ordinal,
        "imc": imc,
        "totalComidasDia": totalComidasDia,
        "puntaje_ia": puntaje_ia,
        "sexo": sexo.lower().strip(),
        "nivel_educativo": nivel_educativo.lower().strip(),
        "estado_imc": estado_imc,
        "estrato": estrato.lower().strip(),
        "inseguridad": inseguridad
    }

    resultado = recomendar(usuario)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "cluster": resultado.get("cluster"),
        "perfil": resultado.get("perfil_resumido"),
        "recomendacion": resultado.get("recomendacion"),
        "usuario": usuario
    })
