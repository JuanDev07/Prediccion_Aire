import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Pronóstico de Calidad del Aire - CORNARE", layout="wide")
st.title("🌬️ Pronóstico de Calidad del Aire (PM10 / PM2.5)")

CARPETA_MODELOS = "modelos_guardados"

# Detección de modelos disponibles
if os.path.exists(CARPETA_MODELOS):
    archivos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith(".pkl")]
else:
    archivos = []

if not archivos:
    st.error("No se encontraron archivos de modelos (.pkl) en la carpeta 'modelos_guardados/'.")
    st.stop()

# Filtros laterales
st.sidebar.header("Configuración del Pronóstico")

estaciones_disponibles = sorted(list(set([f.split("_")[1] for f in archivos])))
estacion_sel = st.sidebar.selectbox("Selecciona la Estación:", estaciones_disponibles)

variables_disponibles = sorted(list(set([f.split("_")[2] for f in archivos if f.split("_")[1] == estacion_sel])))
variable_sel = st.sidebar.selectbox("Selecciona la Variable:", variables_disponibles)

# Filtrar archivos para la selección actual
modelos_estacion = [f for f in archivos if f.startswith(f"modelo_{estacion_sel}_{variable_sel}_")]
modelo_nom_sel = st.sidebar.selectbox("Selecciona el Modelo:", modelos_estacion)

pasos_pronostico = st.sidebar.slider("Horas a pronosticar (a futuro):", min_value=6, max_value=72, value=24, step=6)

# Cargar Modelo seleccionado
ruta_modelo = os.path.join(CARPETA_MODELOS, modelo_nom_sel)
paquete = joblib.load(ruta_modelo)

# Extraer histórico
fechas_hist = pd.to_datetime(paquete["historico"]["fechas"])
valores_hist = paquete["historico"]["valores"]
serie_hist = pd.Series(valores_hist, index=fechas_hist)

# Generar Pronóstico futuro
tipo = paquete["tipo"]
if tipo in ("arima", "sarima", "ses", "holt_winters"):
    pronostico = paquete["modelo"].forecast(pasos_pronostico)
elif tipo == "media_movil":
    ventana = paquete["ventana"]
    hist = list(paquete["ultimos_valores"])
    preds = []
    for _ in range(pasos_pronostico):
        val = np.mean(hist[-ventana:])
        preds.append(val)
        hist.append(val)
    pronostico = pd.Series(preds)
elif tipo == "ventana_deslizante":
    ventana = paquete["tamano_ventana"]
    hist = list(paquete["ultimos_valores"])
    preds = []
    for _ in range(pasos_pronostico):
        in_val = np.array(hist[-ventana:]).reshape(1, -1)
        val = paquete["modelo"].predict(in_val)[0]
        preds.append(val)
        hist.append(val)
    pronostico = pd.Series(preds)

# Construir fechas futuras
ultima_fecha = serie_hist.index[-1]
fechas_futuras = pd.date_range(start=ultima_fecha, periods=pasos_pronostico + 1, freq="h")[1:]
pronostico.index = fechas_futuras

# Mostrar métricas del modelo
meta = paquete.get("metadata", {})
col1, col2, col3 = st.columns(3)
col1.metric("Estación", meta.get("codigo_estacion", estacion_sel))
col2.metric("Modelo", meta.get("nombre_modelo", tipo))
col3.metric("RMSE en Prueba", f"{meta.get('rmse_en_test', 0):.3f}")

# Gráfica
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie_hist.index[-120:], serie_hist.values[-120:], label="Histórico (Últimas 120h)", color="black")
ax.plot(pronostico.index, pronostico.values, label="Pronóstico Futuro", color="crimson", linestyle="--", marker="o", markersize=3)
ax.axvline(ultima_fecha, color="gray", linestyle=":", label="Inicio Pronóstico")
ax.set_ylabel(f"{variable_sel} (µg/m³)")
ax.set_title(f"Pronóstico de {variable_sel} a {pasos_pronostico} horas - Estación {estacion_sel}")
ax.legend()
plt.xticks(rotation=30)
st.pyplot(fig)

# Tabla de resultados
st.subheader("Valores Pronosticados")
df_pred = pd.DataFrame({"Fecha": pronostico.index.strftime("%Y-%m-%d %H:%M:%S"), f"Pronóstico {variable_sel}": pronostico.values})
st.dataframe(df_pred.style.format({f"Pronóstico {variable_sel}": "{:.2f}"}))
