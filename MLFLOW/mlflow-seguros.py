#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Grupo 23 DSA
"""

# Importe de librerías
import numpy as np
import pandas as pd

# Librerías de Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 

#Librerís de Modelado y evaluación
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Lectura de datos
satelitales = pd.read_csv('../Data/Boyaca_NDVI_EVI_SoilMoisture_MENSUAL_2005_2025_CLEAN.csv')
rendimientos = pd.read_excel('../Data/Serie rendimiento cafe 2012 - 2019.xlsx')

# Estandarización y filtrado de NDVI/EVI
df_ndvi = satelitales.copy()
df_ndvi.columns = df_ndvi.columns.str.strip().str.lower()

# Estandarización de fecha
fecha_col = 'fecha' if 'fecha' in df_ndvi.columns else [c for c in df_ndvi.columns if 'time' in c or 'fecha' in c][0]
df_ndvi[fecha_col] = pd.to_datetime(df_ndvi[fecha_col], errors='coerce')

# Slicing datos (Filtro 2012–2019)
df_ndvi = df_ndvi[(df_ndvi[fecha_col].dt.year >= 2012) & (df_ndvi[fecha_col].dt.year <= 2019)]

# Selección de datos
idx_cols = [c for c in df_ndvi.columns if c == 'ndvi' or c == 'evi' or c.startswith('ndvi') or c.startswith('evi')]
df_ndvi = df_ndvi[[fecha_col] + idx_cols].sort_values(fecha_col).reset_index(drop=True)

# Campos auxiliares
df_ndvi['year'] = df_ndvi[fecha_col].dt.year
df_ndvi['month'] = df_ndvi[fecha_col].dt.month
df_ndvi['yyyymm'] = df_ndvi[fecha_col].dt.strftime('%Y-%m')

print("NDVI/EVI listo:", df_ndvi.shape)

# Rendimientos de café en Boyacá (2012-2019)
rend_anual = {
    2012: 0.5,
    2013: 0.9,
    2014: 0.8,
    2015: 0.6,
    2016: 1.0,
    2017: 1.1,
    2018: 1.0,
    2019: 0.9
}

# Normalización del patrón mensual
pattern = np.array([
    0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.08, 0.06, 0.07, 0.10, 0.14, 0.10
])
pattern = pattern / pattern.sum()

# Dataset de rendimientos mensuales
rows = []
for year, total in rend_anual.items():
    for month in range(1, 13):
        fecha = pd.Timestamp(year=year, month=month, day=1)
        rows.append({
            "departamento": "Boyacá",
            "fecha": fecha,
            "anio": year,
            "month": month,
            "yyyymm": fecha.strftime("%Y-%m"),
            "rendimiento_t_ha": round(total * pattern[month - 1], 3)
        })

df_rend_boyaca = pd.DataFrame(rows)

# Validación de proporciones mensuales
df_check = df_rend_boyaca.groupby("anio")["rendimiento_t_ha"].sum().round(3)
print("Comprobación de sumas anuales:")
print(df_check)

#  SIMULACIÓN DE VARIABLES CLIMÁTICAS (Precipitación, Temperatura, Humedad)

fechas = pd.date_range(start="2012-01-01", end="2019-12-31", freq="MS")
np.random.seed(42)  # reproducibilidad

# Simulación climática realista
precipitacion = np.random.normal(loc=160, scale=40, size=len(fechas))  # mm/mes
temp_max = np.random.normal(loc=25, scale=1.5, size=len(fechas))       # °C
temp_min = np.random.normal(loc=15, scale=1.2, size=len(fechas))       # °C
humedad = 100 - (temp_max - 20)*4 + np.random.normal(0, 3, len(fechas))  # relación inversa

df_clima = pd.DataFrame({
    "fecha": fechas,
    "Precipitacion": np.clip(precipitacion, 60, 300),
    "TempMax": np.clip(temp_max, 20, 30),
    "TempMin": np.clip(temp_min, 10, 20),
    "HumedadRelativa": np.clip(humedad, 60, 90)
})

print("Variables climáticas simuladas:", df_clima.shape)

# INTEGRACIÓN 

df_ndvi["yyyymm"]       = df_ndvi[fecha_col].dt.strftime("%Y-%m")
df_rend_boyaca["yyyymm"]= df_rend_boyaca["fecha"].dt.strftime("%Y-%m")
df_clima["yyyymm"]      = df_clima["fecha"].dt.strftime("%Y-%m")

df_modelo = (
    df_ndvi.merge(df_rend_boyaca[["yyyymm","rendimiento_t_ha"]], on="yyyymm", how="inner")
           .merge(df_clima.drop(columns=["fecha"], errors="ignore"), on="yyyymm", how="left")
           .rename(columns={"rendimiento_t_ha":"rendimiento"})
           .assign(fecha=lambda d: pd.to_datetime(d["yyyymm"]))
           .assign(anio=lambda d: d["fecha"].dt.year, month=lambda d: d["fecha"].dt.month)
           .sort_values("fecha", kind="stable")
           .reset_index(drop=True)
)

# Reordenar para visualización rápida
cols = ["fecha","yyyymm","ndvi", "evi", "rendimiento","Precipitacion","TempMax","TempMin","HumedadRelativa"]
df_modelo = df_modelo[[c for c in cols if c in df_modelo.columns]]


# PREPROCESAMIENTO SIMPLE
# Datos para modelado
datos = df_modelo.copy()

# Estandarización de fecha
if "fecha" not in datos.columns:
    if "yyyymm" in datos.columns:
        datos["fecha"] = pd.to_datetime(datos["yyyymm"], format="%Y-%m", errors="coerce")
    else:
        raise ValueError("Falta 'fecha' o 'yyyymm' en df_modelo.")
datos["mes"]  = datos["fecha"].dt.month
datos["anio"] = datos["fecha"].dt.year

# Estandariza nombres NDVI/EVI y clima (tolerante a mayúsculas/minúsculas)
renombres = {}
for c in datos.columns:
    cl = c.strip().lower()
    if cl.startswith("ndvi"): renombres[c] = "NDVI"
    if cl.startswith("evi"):  renombres[c] = "EVI"
    if cl in {"precipitacion"}: renombres[c] = "Precipitacion"
    if cl in {"tmax","tempmax"}: renombres[c] = "TempMax"
    if cl in {"tmin","tempmin"}: renombres[c] = "TempMin"
    if cl in {"humedad","humedadrelativa"}: renombres[c] = "HumedadRelativa"
datos = datos.rename(columns=renombres)

# Estacionalidad mensual (seno/coseno)
datos["mes_sin"] = np.sin(2*np.pi*datos["mes"]/12)
datos["mes_cos"] = np.cos(2*np.pi*datos["mes"]/12)

# Selección de variables (toma solo las que existan)
variables_base = ["NDVI","EVI","Precipitacion","TempMax","TempMin","HumedadRelativa"]
variables_modelo = [v for v in variables_base if v in datos.columns] + ["mes_sin","mes_cos"]

X = datos[variables_modelo].apply(pd.to_numeric, errors="coerce")   # predictores
y = pd.to_numeric(datos["rendimiento"], errors="coerce")            # objetivo

# División temporal y transformación (imputar + escalar)
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

prepro = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler())
])

X_entrena_prep = prepro.fit_transform(X_entrena)
X_prueba_prep  = prepro.transform(X_prueba)

print("Variables usadas:", variables_modelo)
print("Variable objetivo:", "rendimiento")
print("Train:", X_entrena_prep.shape, " | Test:", X_prueba_prep.shape)

#  MODELADO Y EVALUACIÓN 

# Modelos en prueba
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

resultados = []

# Entrenar y evaluar cada modelo
for nombre, modelo in modelos.items():
    modelo.fit(X_entrena_prep, y_entrena)
    y_pred = modelo.predict(X_prueba_prep)
    
    mae = mean_absolute_error(y_prueba, y_pred)
    rmse = np.sqrt(mean_squared_error(y_prueba, y_pred))
    r2 = r2_score(y_prueba, y_pred)
    
    resultados.append([nombre, mae, rmse, r2])

# Tabla de resultados
df_resultados = pd.DataFrame(resultados, columns=["Modelo", "MAE", "RMSE", "R2"]).sort_values("R2", ascending=False)

# Mejor modelo
mejor_modelo = df_resultados.iloc[0, 0]