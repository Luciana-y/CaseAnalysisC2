# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path

# -------------------- Configuración de la página --------------------
st.set_page_config(page_title="Calculadora Inteligente de Envíos", layout="centered")
st.title("Calculadora Inteligente de Envíos")
st.caption("Predice **tiempo_espera** y estima el costo del envío.")

# -------------------- Carga del modelo y features --------------------
@st.cache_resource
def _safe_load_pickle(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_model_and_features():
    here = Path(__file__).resolve().parent

    # 1) Artefactos recomendados: {'model','features'}
    art_path = here / "smart_logistics_artifacts.pkl"
    if art_path.exists():
        art = _safe_load_pickle(art_path)
        return art["model"], list(art["features"])

    # 2) Modelo suelto + reconstrucción de features desde CSV
    for pkl_name in ["linear_regression_model.pkl", "smart_logistics_dataset.pkl"]:
        pkl_path = here / pkl_name
        if pkl_path.exists():
            modelo = _safe_load_pickle(pkl_path)
            csv_path = here / "smart_logistics_dataset.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    "Se encontró el modelo, pero falta 'smart_logistics_artifacts.pkl' "
                    "y el CSV para reconstruir columnas."
                )
            df = pd.read_csv(csv_path)
            mapa = {
                "Timestamp": "marca_tiempo",
                "Asset_ID": "id_activo",
                "Latitude": "latitud",
                "Longitude": "longitud",
                "Inventory_Level": "nivel_inventario",
                "Shipment_Status": "estado_envio",
                "Temperature": "temperatura",
                "Humidity": "humedad",
                "Traffic_Status": "estado_trafico",
                "Waiting_Time": "tiempo_espera",
                "User_Transaction_Amount": "monto_transaccion_usuario",
                "User_Purchase_Frequency": "frecuencia_compra_usuario",
                "Logistics_Delay_Reason": "motivo_retraso_logistico",
                "Asset_Utilization": "utilizacion_activo",
                "Demand_Forecast": "pronostico_demanda",
                "Logistics_Delay": "retraso_logistico",
            }
            df = df.rename(columns=mapa, errors="ignore")
            if "marca_tiempo" in df.columns:
                df["marca_tiempo"] = pd.to_datetime(df["marca_tiempo"], errors="coerce")
            for col in ["estado_envio", "motivo_retraso_logistico"]:
                if col in df.columns:
                    df = df.drop(columns=col)
            objetivo = "tiempo_espera"
            features = [c for c in df.columns
                        if c != objetivo and pd.api.types.is_numeric_dtype(df[c])]
            var0 = [c for c in features if df[c].nunique(dropna=False) <= 1]
            features = [c for c in features if c not in var0]
            return modelo, features

    raise FileNotFoundError(
        "No encontré archivos de modelo. Coloca 'smart_logistics_artifacts.pkl' o un .pkl de modelo junto al CSV."
    )

try:
    modelo, FEATURES = load_model_and_features()
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"No pude cargar el modelo/artefactos: {e}")
    st.stop()

# -------------------- Lógica de tarifas (fijas, según tipo) --------------------
# Ajusta estos valores según tu política
BASE_TARIFAS = {
    "Regular": 1.00,       # S/ por unidad de tiempo del modelo (p.ej., horas)
    "Prioritario": 1.50,   # S/ por unidad de tiempo del modelo
}
MULT_PRIORITARIO = 1.10     # <- multiplicador fijo (no editable en UI)

def obtener_tarifa_base(tipo: str) -> float:
    return float(BASE_TARIFAS.get(tipo, BASE_TARIFAS["Regular"]))

# -------------------- Parámetros (sin entradas editables de costo) --------------------
st.sidebar.header("Parámetros")
tipo_envio = st.sidebar.selectbox("Tipo de envío", ["Regular", "Prioritario"])

def estimar_costo(tiempo_pred: float, tipo: str) -> tuple[float, float]:
    """
    Devuelve (tarifa_base_aplicada, costo_estimado)
    costo = max(0, tiempo_pred) * tarifa_base(tipo) * (1 o MULT_PRIORITARIO)
    """
    tarifa_base = obtener_tarifa_base(tipo)
    tiempo = float(np.maximum(0.0, tiempo_pred))
    factor = MULT_PRIORITARIO if tipo == "Prioritario" else 1.0
    costo = tiempo * tarifa_base * factor
    return tarifa_base, costo

# -------------------- UI de entrada (1 registro) --------------------
def build_single_input(features):
    with st.sidebar.form("form_prediccion", clear_on_submit=False):
        valores = {f: st.number_input(f, value=0.0, step=0.1, format="%.4f")
                   for f in features}
        submitted = st.form_submit_button("Predecir")
    X = pd.DataFrame([valores], columns=features).astype("float64")
    return X, submitted

X_new, submit = build_single_input(FEATURES)

if submit:
    try:
        pred = float(modelo.predict(X_new)[0])
        tarifa_base, costo = estimar_costo(pred, tipo_envio)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tiempo estimado", f"{pred:.2f}")
        c2.metric("Tarifa base aplicada", f"S/ {tarifa_base:.2f}")
        c3.metric("Costo estimado", f"S/ {costo:.2f}")

        st.write("**Entrada usada:**")
        st.dataframe(X_new)
        if tipo_envio == "Prioritario":
            st.caption(f"Tipo: {tipo_envio} · Tarifa base: S/ {tarifa_base:.2f} · Multiplicador fijo: x{MULT_PRIORITARIO:.2f}")
        else:
            st.caption(f"Tipo: {tipo_envio} · Tarifa base: S/ {tarifa_base:.2f}")
    except Exception as e:
        st.error(f"No se pudo realizar la predicción: {e}")

st.divider()

# -------------------- Predicción por lote --------------------
st.subheader("Predicción por lote")
uploaded = st.file_uploader("Sube un CSV con **exactamente** estas columnas:", type="csv")
st.code(", ".join(FEATURES), language="text")

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        faltan = [c for c in FEATURES if c not in df_up.columns]
        if faltan:
            st.error(f"Faltan columnas en tu archivo: {faltan}")
        else:
            Xb = df_up[FEATURES].astype("float64")
            yhat = modelo.predict(Xb).astype(float)

            tarifa_base = obtener_tarifa_base(tipo_envio)
            factor = MULT_PRIORITARIO if tipo_envio == "Prioritario" else 1.0
            costo_est = np.maximum(0.0, yhat) * tarifa_base * factor

            out = df_up.copy()
            out["tiempo_espera_predicho"] = yhat
            out["tarifa_base_aplicada"] = tarifa_base
            out["costo_estimado"] = costo_est

            st.success("Predicción realizada.")
            nota = f"Tipo: {tipo_envio} · Tarifa base: S/ {tarifa_base:.2f}"
            if tipo_envio == "Prioritario":
                nota += f" · Multiplicador fijo: x{MULT_PRIORITARIO:.2f}"
            st.caption(f"Aplicado a todo el archivo → {nota}")
            st.dataframe(out.head(50))

            st.download_button(
                "Descargar predicciones (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_envios.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error leyendo o procesando el CSV: {e}")

with st.expander("Detalles del modelo"):
    try:
        coefs = getattr(modelo[-1], "coef_", None)
        if coefs is not None and len(coefs) == len(FEATURES):
            st.write("Coeficientes (orden = features):")
            st.dataframe(pd.DataFrame({"feature": FEATURES, "coef": coefs}).sort_values("coef"))
        st.write("Pipeline:", modelo)
    except Exception:
        pass

with st.expander("Lógica de tarifas (editable por código)"):
    st.write("Tarifas base aplicadas por tipo de envío y multiplicador fijo para prioritario:")
    st.dataframe(pd.DataFrame(
        [{"tipo_envio": k, "tarifa_base": v} for k, v in BASE_TARIFAS.items()]
    ))
    st.write(f"Multiplicador prioritario fijo: x{MULT_PRIORITARIO:.2f}")
    st.caption("Edita los dict/constantes BASE_TARIFAS y MULT_PRIORITARIO en el código para cambiar estos valores.")