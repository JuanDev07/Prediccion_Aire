import os
import requests
import urllib3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuración
ESTACIONES = ["401", "402"]  # Códigos de las 2 estaciones
VARIABLES = ["PM10", "PM25"]
HORAS_TEST = 48
PERIODO_ESTACIONAL = 24
CARPETA_MODELOS = "modelos_guardados"
API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(CARPETA_MODELOS, exist_ok=True)

def obtener_datos(estacion, variable):
    url = f"{API_BASE_URL}/{estacion}/{variable}"
    params = {"desde": "2026-08-08", "hasta": "2026-09-08", "calidad": 1}
    
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30, verify=False)
        datos = resp.json()
        registros = list(datos.get("values", []))
        siguiente = datos.get("next")
        
        while siguiente:
            r = requests.get(siguiente, timeout=30, verify=False, headers=HEADERS)
            data_pag = r.json()
            registros.extend(data_pag.get("values", []))
            siguiente = data_pag.get("next")
            
        df = pd.DataFrame(registros).rename(columns={"fecha": "fecha", "muestra": "valor"})
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.sort_values("fecha").drop_duplicates(subset="fecha").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error consultando estación {estacion} ({variable}): {e}")
        return None

def limpiar_datos(df):
    df_idx = df.set_index("fecha")
    rango = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq="h")
    df_reg = df_idx.reindex(rango)
    df_reg["valor"] = df_reg["valor"].interpolate(method="time").ffill().bfill()
    
    # Outliers IQR + físico
    Q1, Q3 = df_reg["valor"].quantile(0.25), df_reg["valor"].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_reg["valor"] < (Q1 - 1.5 * IQR)) | (df_reg["valor"] > (Q3 + 1.5 * IQR)) | (df_reg["valor"] < 0)
    df_reg.loc[mask, "valor"] = np.nan
    df_reg["valor"] = df_reg["valor"].interpolate(method="time").ffill().bfill()
    return df_reg["valor"]

def procesar_estacion(estacion, variable):
    print(f"\n--- Procesando Estación: {estacion} | Variable: {variable} ---")
    df = obtener_datos(estacion, variable)
    if df is None or len(df) < 50:
        print(f"Insuficientes datos para la estación {estacion}.")
        return

    serie = limpiar_datos(df)
    test_len = min(HORAS_TEST, len(serie) // 5)
    train, test = serie.iloc[:-test_len], serie.iloc[-test_len:]

    # Evaluamos ARIMA (0..2)
    best_rmse, best_order, best_arima = float("inf"), (1, 0, 1), None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(train, order=(p, d, q)).fit()
                    p_val = m.forecast(test_len)
                    rmse = np.sqrt(mean_squared_error(test, p_val))
                    if rmse < best_rmse:
                        best_rmse, best_order = rmse, (p, d, q)
                except:
                    continue

    # Entrenar modelo final con TODA la serie
    modelo_arima = ARIMA(serie, order=best_order).fit()
    
    paquete_arima = {
        "tipo": "arima",
        "modelo": modelo_arima,
        "orden": best_order,
        "metadata": {
            "nombre_modelo": f"ARIMA{best_order}",
            "variable": variable,
            "codigo_estacion": estacion,
            "rmse_en_test": float(best_rmse),
            "fecha_entrenamiento": datetime.now().isoformat()
        },
        "historico": {"fechas": serie.index.astype(str).tolist(), "valores": serie.values.tolist()}
    }
    
    ruta_arima = f"{CARPETA_MODELOS}/modelo_{estacion}_{variable}_ARIMA.pkl"
    joblib.dump(paquete_arima, ruta_arima)

    # Entrenar SES
    modelo_ses = SimpleExpSmoothing(serie, initialization_method="estimated").fit()
    rmse_ses = np.sqrt(mean_squared_error(test, SimpleExpSmoothing(train, initialization_method="estimated").fit().forecast(test_len)))
    
    paquete_ses = {
        "tipo": "ses",
        "modelo": modelo_ses,
        "metadata": {
            "nombre_modelo": "SES",
            "variable": variable,
            "codigo_estacion": estacion,
            "rmse_en_test": float(rmse_ses),
            "fecha_entrenamiento": datetime.now().isoformat()
        },
        "historico": {"fechas": serie.index.astype(str).tolist(), "valores": serie.values.tolist()}
    }
    
    ruta_ses = f"{CARPETA_MODELOS}/modelo_{estacion}_{variable}_SES.pkl"
    joblib.dump(paquete_ses, ruta_ses)
    print(f"Modelos guardados para estación {estacion} ({variable}): {ruta_arima}, {ruta_ses}")

if __name__ == "__main__":
    for est in ESTACIONES:
        for var in VARIABLES:
            procesar_estacion(est, var)
