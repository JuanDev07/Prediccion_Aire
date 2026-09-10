import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Pronóstico de Calidad del Aire - CORNARE", layout="wide")
st.title("🌬️ Pronóstico de Calidad del Aire (PM10 / PM2.5)")

CARPETA_MODELOS = "modelos_guardados"

if os.path.exists(CARPETA_MODELOS):
    archivos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith(".pkl")]
else:
    archivos = []

if not archivos:
    st.error("No se encontraron archivos de modelos (.pkl) en la carpeta 'modelos_guardados/'.")
    st.stop()

st.sidebar.header("Configuración del Pronóstico")

estaciones_disponibles = sorted(list(set([f.split("_")[1] for f in archivos])))
estacion_sel = st.sidebar.selectbox("Selecciona la Estación:", estaciones_disponibles)

variables_disponibles = sorted(list(set([f.split("_")[2] for f in archivos if f.split("_")[1] == estacion_sel])))
variable_sel = st.sidebar.selectbox("Selecciona la Variable:", variables_disponibles)

modelos_estacion = [f for f in archivos if f.startswith(f"modelo_{estacion_sel}_{variable_sel}_")]
modelo_nom_sel = st.sidebar.selectbox("Selecciona el Modelo:", modelos_estacion)

pasos_pronostico = st.sidebar.slider("Horas a pronosticar (a futuro):", min_value=6, max_value=72, value=24, step=6)

ruta_modelo = os.path.join(CARPETA_MODELOS, modelo_nom_sel)
paquete = joblib.load(ruta_modelo)

fechas_hist = pd.to_datetime(paquete["historico"]["fechas"])
valores_hist = paquete["historico"]["valores"]
serie_hist = pd.Series(valores_hist, index=fechas_hist)

tipo = paquete["tipo"]

# Generación del pronóstico
if tipo == "ses":
    modelo_ses = SimpleExpSmoothing(serie_hist, initialization_method="estimated").fit()
    pronostico = modelo_ses.forecast(pasos_pronostico)
elif tipo == "arima":
    if "modelo" in paquete:
        try:
            pronostico = paquete["modelo"].forecast(pasos_pronostico)
        except Exception:
            orden = paquete.get("orden", (1, 0, 1))
            modelo_arima = ARIMA(serie_hist, order=orden).fit()
            pronostico = modelo_arima.forecast(pasos_pronostico)
    else:
        orden = paquete.get("orden", (1, 0, 1))
        modelo_arima = ARIMA(serie_hist, order=orden).fit()
        pronostico = modelo_arima.forecast(pasos_pronostico)

ultima_fecha = serie_hist.index[-1]
fechas_futuras = pd.date_range(start=ultima_fecha, periods=pasos_pronostico + 1, freq="h")[1:]
pronostico.index = fechas_futuras

meta = paquete.get("metadata", {})
col1, col2, col3 = st.columns(3)
col1.metric("Estación", meta.get("codigo_estacion", estacion_sel))
col2.metric("Modelo", meta.get("nombre_modelo", tipo.upper()))
col3.metric("RMSE en Prueba", f"{meta.get('rmse_en_test', 0):.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie_hist.index[-120:], serie_hist.values[-120:], label="Histórico (Últimas 120h)", color="black")
ax.plot(pronostico.index, pronostico.values, label="Pronóstico Futuro", color="crimson", linestyle="--", marker="o", markersize=3)
ax.axvline(ultima_fecha, color="gray", linestyle=":", label="Inicio Pronóstico")
ax.set_ylabel(f"{variable_sel} (µg/m³)")
ax.set_title(f"Pronóstico de {variable_sel} a {pasos_pronostico} horas - Estación {estacion_sel}")
ax.legend()
plt.xticks(rotation=30)
st.pyplot(fig)

st.subheader("Valores Pronosticados")
df_pred = pd.DataFrame({"Fecha": pronostico.index.strftime("%Y-%m-%d %H:%M:%S"), f"Pronóstico {variable_sel}": pronostico.values})
st.dataframe(df_pred.style.format({f"Pronóstico {variable_sel}": "{:.2f}"}))
