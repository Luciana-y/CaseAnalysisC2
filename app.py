# app.py
# -------------------------------------------------------------
# Calculadora de envíos (versión lista para demo)
# - ETA (regresión lineal)
# - Monto transacción (regresión lineal)
# - Prob. de cumplimiento (regresión logística)
# - Costos didácticos
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
from modelos import LinearRegressionFromScratch, LinearRegressionCostFromScratch, LogisticRegressionFromScratch

# =========================
# Configuraciones
# =========================
FEATURES_NUM = ['Temperature','Humidity','Inventory_Level','Asset_Utilization','Demand_Forecast','User_Purchase_Frequency']
CATEGORICAL_ETA = ["estado_envio", "estado_trafico", "motivo_retraso_logistico"]
CATEGORICAL_DELAY = ["estado_trafico"]

# Mapeo de nombres (entrenamiento ↔ interfaz)
NAME_MAP = {
    "Temperature": "temperatura",
    "Humidity": "humedad",
    "Inventory_Level": "nivel_inventario",
    "Asset_Utilization": "utilizacion_activo",  # corregido
    "Demand_Forecast": "pronostico_demanda",
    "User_Purchase_Frequency": "frecuencia_compra_usuario"
}
# Invertido (español → inglés)
NAME_MAP_INV = {v: k for k, v in NAME_MAP.items()}

# Fórmula de costo
def calcular_costo(distancia_km, tipo_envio, demanda):
    base = {"Express": 7, "Standard": 5, "Economy": 3}
    rate = {"Express": 0.9, "Standard": 0.6, "Economy": 0.4}
    return base[tipo_envio] + rate[tipo_envio]*distancia_km * (1 + 0.1*demanda)

# =========================
# Carga de modelos y preprocessors
# =========================
MODELOS_DIR = os.path.join(os.path.dirname(__file__), "Datos guardados")

reg = load(os.path.join(MODELOS_DIR, "reg.joblib"))           # ETA
reg_cost = load(os.path.join(MODELOS_DIR, "reg_cost.joblib")) # Monto
logreg = load(os.path.join(MODELOS_DIR, "logreg.joblib"))     # Retraso

scaler_eta = load(os.path.join(MODELOS_DIR, "scaler_eta.joblib"))
encoder_eta = load(os.path.join(MODELOS_DIR, "encoder_eta.joblib"))

scaler_cost = load(os.path.join(MODELOS_DIR, "scaler_cost.joblib"))
encoder_cost = load(os.path.join(MODELOS_DIR, "encoder_cost.joblib"))

preprocessor_delay = load(os.path.join(MODELOS_DIR, "preprocessor_delay.joblib"))

# =========================
# UI – Entradas
# =========================
st.set_page_config(page_title="Calculadora de Envíos", layout="wide")
st.title("📦 Calculadora de Envíos — demo educativa")
st.caption("ETA + Prob. de cumplimiento + Monto transacción + Costos")

st.sidebar.header("⚙️ Parámetros de entrada")

# Numéricas
distance_km = st.sidebar.slider("Distancia estimada (km)", 1, 200, 25)
demanda_nivel = st.sidebar.select_slider("Nivel de demanda", options=["Baja","Media","Alta"], value="Media")
demanda_factor = {"Baja": 0, "Media": 1, "Alta": 2}[demanda_nivel]

temp = st.sidebar.slider("Temperatura (°C)", -5, 45, 25)
hum  = st.sidebar.slider("Humedad (%)", 10, 100, 60)
inv  = st.sidebar.slider("Inventario (0-1000)", 0, 1000, 300)
util = st.sidebar.slider("Uso de activos (%)", 0, 100, 70)
dfor = st.sidebar.slider("Demanda forecast (0-1000)", 0, 1000, 250)

# Categóricas
estado_envio = st.sidebar.selectbox("Estado del envío", ["En tránsito", "Pendiente", "Entregado"])
estado_trafico = st.sidebar.selectbox("Estado del tráfico", ["Normal", "Congestionado", "Accidente"])
motivo_retraso = st.sidebar.selectbox("Motivo del retraso logístico", ["Ninguno", "Clima", "Demanda alta", "Otro"])

contexto = st.sidebar.radio("Contexto", ["Económico", "Urgente"], index=0)

# Obtener el orden exacto con el que fue entrenado
FEATURES_NUM = list(scaler_eta.feature_names_in_)
# Entrada del usuario
X_num = pd.DataFrame([{
    "temperatura": temp,
    "humedad": hum,
    "nivel_inventario": inv,
    "utilizacion_activo": util,
    "pronostico_demanda": dfor,
    "frecuencia_compra_usuario": 1
}])

# Reordenar con el orden original del scaler
X_num = X_num[scaler_eta.feature_names_in_]

# Transformar
X_num_scaled = scaler_eta.transform(X_num)
X_cat_dummy = encoder_eta.transform(pd.DataFrame([{
    "estado_envio": estado_envio,
    "estado_trafico": estado_trafico,
    "motivo_retraso_logistico": motivo_retraso
}]))
X_all_eta = np.hstack([X_num_scaled, X_cat_dummy])
X_all_eta = np.c_[np.ones((X_all_eta.shape[0],1)), X_all_eta]  # bias

eta_min = float(reg.predict(X_all_eta)[0])
eta_min = max(1.0, eta_min)
eta_low, eta_high = eta_min*0.8, eta_min*1.2

# Monto transacción
X_num_cost = scaler_cost.transform(X_num)
X_cat_cost = encoder_cost.transform(pd.DataFrame([{
    "estado_envio": estado_envio,
    "estado_trafico": estado_trafico,
    "motivo_retraso_logistico": motivo_retraso
}])).toarray()
X_all_cost = np.hstack([X_num_cost, X_cat_cost])

monto = float(reg_cost.predict(X_all_cost)[0])
monto = max(0.0, monto)

# Prob. de retraso
X_delay = preprocessor_delay.transform(pd.DataFrame([{
    "temperatura": temp,
    "humedad": hum,
    "nivel_inventario": inv,
    "utilizacion_activo": util,
    "pronostico_demanda": dfor,
    "frecuencia_compra_usuario": 1,   # <--- agregado
    "estado_trafico": estado_trafico
}]))

if hasattr(X_delay, "toarray"):
    X_delay = X_delay.toarray()

prob_delay = float(logreg.predict_proba(X_delay)[0,1])
prob_delay = min(max(prob_delay, 0.0), 1.0)
prob_ok = 1 - prob_delay

# Costos
cost_express  = calcular_costo(distance_km, "Express",  demanda_factor)
cost_standard = calcular_costo(distance_km, "Standard", demanda_factor)
cost_economy  = calcular_costo(distance_km, "Economy",  demanda_factor)

# =========================
# Salida
# =========================
st.subheader("Resultados")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("### ⏱️ ETA")
    st.metric("ETA central", f"{eta_min:.1f} min")
    st.caption(f"Rango: {eta_low:.1f}–{eta_high:.1f} min (±20%)")

with c2:
    st.markdown("### 🎯 Cumplimiento")
    st.metric("Prob. de cumplimiento", f"{prob_ok*100:.1f}%")
    st.progress(prob_ok)

with c3:
    st.markdown("### 💰 Monto transacción")
    st.metric("Monto estimado", f"${monto:.2f}")

with c4:
    st.markdown("### 💸 Costos")
    st.write(f"Express:  ${cost_express:.2f}")
    st.write(f"Standard: ${cost_standard:.2f}")
    st.write(f"Economy:  ${cost_economy:.2f}")

# Tabla comparativa
rows = [
    {"Opción": "Express",  "Costo (USD)": cost_express,  "ETA (min)": eta_min*0.9, "Cumplimiento (%)": prob_ok*100*1.05},
    {"Opción": "Standard", "Costo (USD)": cost_standard, "ETA (min)": eta_min*1.0, "Cumplimiento (%)": prob_ok*100*1.00},
    {"Opción": "Economy",  "Costo (USD)": cost_economy,  "ETA (min)": eta_min*1.1, "Cumplimiento (%)": prob_ok*100*0.95},
]
tbl = pd.DataFrame(rows)
tbl["ETA (min)"] = tbl["ETA (min)"].clip(lower=1).round(1)
tbl["Cumplimiento (%)"] = tbl["Cumplimiento (%)"].clip(lower=5, upper=99).round(1)
tbl["Costo (USD)"] = tbl["Costo (USD)"].round(2)

if contexto == "Urgente":
    tbl = tbl.sort_values(by=["ETA (min)", "Cumplimiento (%)"], ascending=[True, False])
else:
    tbl = tbl.sort_values(by=["Costo (USD)", "Cumplimiento (%)"], ascending=[True, False])

st.markdown("### 🔍 Comparación de opciones")
st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

with st.expander("ℹ️ ¿Cómo se calculó?"):
    st.write("- ETA: regresión lineal sobre variables operativas.")
    st.write("- Rango ETA: ±20% (aprox. didáctica).")
    st.write("- Cumplimiento: 1 − prob(retraso) de la regresión logística.")
    st.write("- Monto: regresión lineal sobre mismas variables.")
    st.write("- Costo: base + tarifa/km × (1 + 0.1×demanda).")
    st.write("- Orden según contexto: Urgente (menor ETA) vs Económico (menor costo).")
