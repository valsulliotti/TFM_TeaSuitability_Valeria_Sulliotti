"""
Aplicación TeaSuitability con datos reales - Evaluador de idoneidad del té.

Interfaz web (Streamlit) que carga el modelo entrenado con datos reales,
recibe las condiciones de un lugar y devuelve su idoneidad, una explicación
con SHAP y un informeo.

Ejecución: streamlit run TeaSuitability_App.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# clases necesarias para poder cargar el modelo guardado (.pkl)
from lightgbm import LGBMClassifier  # noqa: F401
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.pipeline import make_pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.svm import SVC  # noqa: F401


st.set_page_config(page_title="TeaSuitability", page_icon="🍵", layout="centered")

# Rutas (ajusta CARPETA_BASE si ejecutas en local)
if Path("/content/drive/MyDrive/TFM_TeaSuitability").exists():
    CARPETA_BASE = Path("/content/drive/MyDrive/TFM_TeaSuitability")  # Colab
else:
    CARPETA_BASE = Path(".")  # local: junto a TeaSuitability_App.py
RUTA_MODELO = CARPETA_BASE / "modelo_teasuitability_real.pkl"
RUTA_CSV = CARPETA_BASE / "dataset_master.csv"

# Las 8 variables del modelo (mismo orden que en el entrenamiento del B.3)
COLUMNAS = [
    "temperatura_media",        # bio1
    "rango_diurno",             # bio2
    "precipitacion_anual",      # bio12
    "estacionalidad_precip",    # bio15
    "precip_trimestre_seco",    # bio17
    "elevacion",
    "ph_suelo",
]


@st.cache_resource
def cargar_modelo(ruta):
    """Carga el modelo entrenado (cacheado para no recargarlo en cada acción)."""
    return joblib.load(ruta)


@st.cache_data
def cargar_fondo(ruta):
    """Carga los datos de referencia (fondo) para SHAP."""
    return pd.read_csv(ruta)[COLUMNAS]


def probabilidad_a_banda(probabilidad):
    """Traduce una probabilidad (0-1) a banda de idoneidad."""
    if probabilidad < 0.25:
        return "No apta"
    if probabilidad < 0.50:
        return "Baja"
    if probabilidad < 0.75:
        return "Media"
    return "Alta"


def generar_informe(probabilidad, banda, contribuciones):
    """Redacta un informe en lenguaje natural explicando la idoneidad."""
    ordenadas = sorted(contribuciones.items(), key=lambda x: abs(x[1]), reverse=True)
    lineas = [
        f"Resultado: idoneidad {banda.upper()} "
        f"(probabilidad estimada: {probabilidad:.1%}).",
        "",
        "Factores que más influyen:",
    ]
    for variable, valor in ordenadas:
        efecto = "FAVORECE la idoneidad" if valor >= 0 else "REDUCE la idoneidad"
        lineas.append(f"  - {variable}: {efecto} (peso {abs(valor):.3f}).")
    lineas.append("")
    lineas.append(
        f"En resumen, el factor más determinante es '{ordenadas[0][0]}'. "
        f"El sistema clasifica este lugar como de idoneidad {banda.lower()}."
    )
    return "\n".join(lineas)


# Interfaz
st.title("🍵 TeaSuitability")
st.caption(
    "Evaluador de idoneidad geográfica para el cultivo de té de calidad "
    "(modelo entrenado con datos reales de GBIF, WorldClim y SoilGrids)."
)

if not RUTA_MODELO.exists():
    st.error(f"No encuentro el modelo en {RUTA_MODELO}. Ejecuta antes el B.3.")
    st.stop()

modelo = cargar_modelo(str(RUTA_MODELO))
fondo_datos = cargar_fondo(str(RUTA_CSV))

# Entradas del usuario (barra lateral) con valores por defecto = perfil del té
st.sidebar.header("Condiciones del lugar")
st.sidebar.caption("Rangos óptimos según la literatura agronómica del té.")
temp = st.sidebar.slider(
    "Temperatura media anual (°C)", 0.0, 35.0, 20.0, 0.5,
    help="Óptimo 18-25 °C; el crecimiento se reduce por debajo de 13 °C y por "
         "encima de 30 °C (Hajiboland, 2017).")
rango = st.sidebar.slider(
    "Rango diurno medio (°C)", 3.0, 20.0, 10.0, 0.5,
    help="Diferencia media entre la máxima del día y la mínima nocturna; las "
         "noches frescas favorecen la calidad del té.")
precip = st.sidebar.slider(
    "Precipitación anual (mm)", 500, 5000, 2000, 50,
    help="Mínimo ~1200 mm; óptimo 1500-3000 mm. La sequía es el principal "
         "factor limitante (Hajiboland, 2017; Bania et al., 2025).")
est_precip = st.sidebar.slider(
    "Estacionalidad de precipitación", 5.0, 185.0, 50.0, 1.0,
    help="Variación de la lluvia entre estaciones; el té prefiere lluvia bien "
         "distribuida durante el año.")
precip_seco = st.sidebar.slider(
    "Precipitación del trimestre seco (mm)", 0, 900, 200, 10,
    help="Lluvia en la estación seca; poca lluvia genera estrés hídrico, que "
         "reduce el rendimiento del té.")
elev = st.sidebar.slider(
    "Elevación (m)", 0, 3000, 900, 20,
    help="El té se cultiva de 0 a ~2200 m; las altitudes medias-altas dan té "
         "de mayor calidad.")
ph = st.sidebar.slider(
    "pH del suelo (ácido = mejor para el té)", 3.5, 8.5, 5.0, 0.1,
    help="El té requiere suelo ácido; óptimo 4.5-5.5. Por encima de 6.0 la "
         "idoneidad disminuye (Bania et al., 2025).")

if st.sidebar.button("Evaluar idoneidad", type="primary"):
    # construir el punto con las condiciones introducidas
    punto = pd.DataFrame([{
        "temperatura_media": temp,
        "rango_diurno": rango,
        "precipitacion_anual": precip,
        "estacionalidad_precip": est_precip,
        "precip_trimestre_seco": precip_seco,
        "elevacion": elev,
        "ph_suelo": ph,
    }])[COLUMNAS]   # asegurar el mismo orden que en el entrenamiento

    probabilidad = modelo.predict_proba(punto)[0, 1]
    banda = probabilidad_a_banda(probabilidad)

    col1, col2 = st.columns(2)
    col1.metric("Probabilidad de idoneidad", f"{probabilidad:.1%}")
    col2.metric("Banda de idoneidad", banda)

    with st.spinner("Calculando la explicación..."):
        fondo = shap.kmeans(fondo_datos, 10)
        explicador = shap.KernelExplainer(
            lambda z: modelo.predict_proba(z)[:, 1], fondo)
        valores_shap = explicador.shap_values(punto, nsamples=100)
        contribuciones = dict(zip(COLUMNAS, valores_shap[0]))

    st.subheader("¿Por qué este resultado?")
    orden = sorted(contribuciones.items(), key=lambda x: x[1])
    nombres = [n for n, _ in orden]
    valores = [v for _, v in orden]
    colores = ["#c0392b" if v < 0 else "#27ae60" for v in valores]
    figura, eje = plt.subplots(figsize=(7, 3.5))
    eje.barh(nombres, valores, color=colores)
    eje.axvline(0, color="black", linewidth=0.8)
    eje.set_xlabel("Contribución (verde favorece, rojo reduce)")
    st.pyplot(figura)

    st.subheader("Informe")
    st.text(generar_informe(probabilidad, banda, contribuciones))
else:
    st.info("Ajusta las condiciones en la barra lateral y pulsa 'Evaluar idoneidad'.")