# ========================================================
# app_seguro_indexado.py — Dashboard integrado FINAL
# ========================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer

# ========================================================
# RUTA BASE
# ========================================================
BASE_DIR = Path(__file__).resolve().parent
ARTI = BASE_DIR / "artifacts"
DATA = BASE_DIR / "Data"

# ========================================================
# CARGA MODELO Y DATASET
# ========================================================
DF = pd.read_csv(ARTI / "dataset_modelo.csv")
META = json.loads((ARTI / "metadata.json").read_text(encoding="utf-8"))
PIPE = joblib.load(ARTI / "modelo_boyaca.pkl")
FEATURE_COLS = joblib.load(ARTI / "feature_cols.pkl")

# ========================================================
# CARGA GEOJSON REAL DE COLOMBIA
# ========================================================
GEO_COL = json.loads((DATA / "colombia.geo.json").read_text(encoding="utf-8"))

# ========================================================
# FIX IMPUTER (VERSIÓN 1.5 → 1.3)
# ========================================================
def fix_imputer_dtype(est):
    if isinstance(est, SimpleImputer):
        if getattr(getattr(est, "_fit_dtype", None), "kind", None) is None:
            est._fit_dtype = np.dtype("float64")
    if hasattr(est, "named_steps"):
        for s in est.named_steps.values():
            fix_imputer_dtype(s)

fix_imputer_dtype(PIPE)

# ========================================================
# NORMALIZACIÓN FECHAS
# ========================================================
if "fecha" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["fecha"], errors="coerce")
elif "yyyymm" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["yyyymm"], format="%Y-%m", errors="coerce")

DF["rendimiento"] = pd.to_numeric(DF["rendimiento"], errors="coerce")
DF = DF.dropna(subset=["fecha", "rendimiento"]).copy()
DF = DF.sort_values("fecha")

DF["anio"] = DF["fecha"].dt.year
DF["mes_num"] = DF["fecha"].dt.month
DF["mes_nombre"] = DF["mes_num"].map({
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"
})

CAND_VARS = [
    v for v in ["NDVI","EVI","Precipitacion","TempMax","TempMin","HumedadRelativa"]
    if v in DF.columns
]

# ========================================================
# FUNCIÓN KPI CARD (AGREGADA Y CORREGIDA)
# ========================================================
def kpi_card(title, value, subtitle=None, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H4(value, className="card-title"),
            html.Div(subtitle or "", className="text-muted small"),
        ]),
        className=f"shadow-sm border-0 bg-{color} bg-opacity-10",
        style={"height": "100%"}
    )

# ========================================================
# FUNCIONES AUXILIARES
# ========================================================
ALIASES = {
    "ndvi":["ndvi"], "evi":["evi"],
    "precipitacion":["prec","precipitacion","rain","ppt","mm"],
    "tempmax":["tmax","tempmax"], "tempmin":["tmin","tempmin"],
    "humedadrelativa":["hum","rh","humedadrelativa"],
    "month_sin":["month_sin","mes_sin"],
    "month_cos":["month_cos","mes_cos"],
    "mes":["mes","month"], "anio":["anio","year"],
    "fecha":["fecha","date","yyyymm","yearmonth"]
}

def build_pred_row(ndvi,evi,prec,tmax,tmin,hum,mes,req_cols):
    m_sin=np.sin(2*np.pi*mes/12)
    m_cos=np.cos(2*np.pi*mes/12)
    vals = {
        "ndvi":float(ndvi),
        "evi":float(evi),
        "precipitacion":float(prec),
        "tempmax":float(tmax),
        "tempmin":float(tmin),
        "humedadrelativa":float(hum),
        "month_sin":m_sin,
        "month_cos":m_cos,
        "mes":int(mes),
        "anio":int(DF["anio"].max()),
        "fecha":int(DF["anio"].max()*100+mes)
    }

    def match(col):
        cl=col.lower()
        if cl in vals: return vals[cl]
        for k,als in ALIASES.items():
            if any(a in cl for a in als) and k in vals:
                return vals[k]
        return np.nan

    row = pd.DataFrame([{c:match(c) for c in req_cols}], columns=req_cols)
    return row.apply(pd.to_numeric, errors="coerce")

# ========================================================
# RANGOS PARA SLIDERS
# ========================================================
def get_range(col, dmin, dmax):
    if col in DF:
        a=float(DF[col].min())
        b=float(DF[col].max())
        m=float(DF[col].median())
        return a,b,m
    return dmin,dmax,(dmin+dmax)/2

def make_marks(vmin, vmax, n=5, digits=0):
    vals=np.linspace(vmin, vmax, n)
    return {
        round(v,digits):{
            "label":str(round(v,digits)),
            "style":{"fontSize":"12px"}
        }
        for v in vals
    }

# NDVI – EVI
ndvi_min,ndvi_max,ndvi_med=get_range("NDVI",0.2,0.95)
evi_min,evi_max,evi_med=get_range("EVI",0.1,0.9)

ndvi_marks=make_marks(ndvi_min,ndvi_max,5,2)
evi_marks=make_marks(evi_min,evi_max,5,2)

# Precipitación
prec_min,prec_max,prec_med=get_range("Precipitacion",60,300)
prec_marks=make_marks(prec_min,prec_max,6,0)

# Temperaturas
tmax_min,tmax_max,tmax_med=get_range("TempMax",20,30)
tmin_min,tmin_max,tmin_med=get_range("TempMin",10,20)

tmax_marks=make_marks(tmax_min,tmax_max,5,1)
tmin_marks=make_marks(tmin_min,tmin_max,5,1)

# Humedad
hum_min,hum_max,hum_med=get_range("HumedadRelativa",60,90)
hum_marks=make_marks(hum_min,hum_max,5,0)

# ========================================================
# FIGURAS BASE PRINCIPALES (sin tocar)
# ========================================================
fig_ts_base = px.line(
    DF,x="fecha",y="rendimiento",
    labels={"fecha":"Fecha","rendimiento":"t/ha"}
).update_layout(margin=dict(l=10,r=10,t=10,b=10))

fig_season = px.box(
    DF,x="mes_nombre",y="rendimiento",
    category_orders={"mes_nombre":[
        "Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"
    ]},
    labels={"mes_nombre":"Mes","rendimiento":"t/ha"},
    title="Estacionalidad mensual"
).update_layout(margin=dict(l=10,r=10,t=40,b=10))

# ========================================================
# MÉTRICAS DEL MODELO
# ========================================================
metricas = META.get("metricas_test_snapshot") or META.get("metricas_notebook") or {}
R2_val = metricas.get("R2", 0.0)
RMSE_val = metricas.get("RMSE", 0.0)
MAE_val = metricas.get("MAE", 0.0)
N_obs = META.get("n_obs", len(DF))


# ========================================================
# DASH APP
# ========================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Dashboard Boyacá")


# ========================================================
# ------------  FUNCIÓN DE RIESGO MULTIVARIADO ------------
# ========================================================
def riesgo_multivariado(ndvi, evi, prec, tmax, tmin, hum):
    # Cuantiles reales desde DF
    p25_ndvi = DF["NDVI"].quantile(0.25)
    p25_evi  = DF["EVI"].quantile(0.25)
    p25_prec = DF["Precipitacion"].quantile(0.25)
    p75_prec = DF["Precipitacion"].quantile(0.75)

    # --- A. Vegetación ---
    if ndvi < p25_ndvi and evi < p25_evi:
        Rv = 1.0
    elif ndvi < p25_ndvi:
        Rv = 0.7
    elif evi < p25_evi:
        Rv = 0.6
    elif evi < ndvi - 0.1:   # contradicción
        Rv = 0.5
    elif ndvi < 0.55:
        Rv = 0.2
    else:
        Rv = 0.0

    # --- B. Lluvia ---
    if prec > p75_prec:
        Rl = 1.0
    elif prec < p25_prec:
        Rl = 0.7
    else:
        Rl = 0.3

    # --- C. Temperatura ---
    if tmax > 29 and tmin > 18:
        Rt = 1.0
    elif tmax > 28:
        Rt = 0.6
    else:
        Rt = 0.3

    # --- D. Humedad ---
    if hum < 60:
        Rh = 1.0
    elif hum > 90:
        Rh = 0.6
    else:
        Rh = 0.3

    # --- COMBINACIÓN ---
    riesgo = 0.35*Rv + 0.30*Rl + 0.20*Rt + 0.15*Rh
    return round(float(riesgo), 3)


# ========================================================
# TABLA DE COLORES DEL RIESGO
# ========================================================
def color_riesgo(r):
    if r > 0.75: return "#d62728"   # rojo – muy alto
    if r > 0.60: return "#ff7f0e"   # naranja – alto
    if r > 0.40: return "#f2d600"   # amarillo – medio
    if r > 0.20: return "#1f77b4"   # azul – bajo
    return "#2ca02c"                # verde – muy bajo

def clasif_riesgo(r):
    """Devuelve la categoría de riesgo como texto."""
    if r > 0.75:
        return "Muy alto"
    elif r > 0.60:
        return "Alto"
    elif r > 0.40:
        return "Medio"
    elif r > 0.20:
        return "Bajo"
    else:
        return "Muy bajo"

# ========================================================
# LAYOUT PRINCIPAL (TAL COMO EL TUYO, SIN CAMBIOS)
# ========================================================
app.layout = dbc.Container(fluid=True, children=[

    html.Br(),
    dbc.Row([
        dbc.Col(html.H2("📊 Dashboard – Rendimiento de Café (Boyacá)"), md=8),
        dbc.Col(html.Div(f"Modelo: {META.get('model', 'RandomForestRegressor')}"),
                md=4, className="text-end align-self-center")
    ]),
    html.Hr(),

    # ----------------- KPIs -----------------
    dbc.Row([
        dbc.Col(kpi_card("R² (test)", f"{R2_val:.3f}", color="success"), md=3),
        dbc.Col(kpi_card("RMSE (test)", f"{RMSE_val:.3f}", color="danger"),  md=3),
        dbc.Col(kpi_card("MAE (test)", f"{MAE_val:.3f}", color="warning"),   md=3),
        dbc.Col(kpi_card("Observaciones", f"{N_obs:,}"), md=3),
    ], className="g-3 mb-3"),

    # ============================================================
    #          PRIMERA FILA : Predicción + Serie histórica
    # ============================================================
    dbc.Row([
        # -------- COLUMNA IZQUIERDA: SLIDERS --------
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Predicción interactiva", className="mb-3"),

                # ------------ Mes ------------
                dbc.Row([
                    dbc.Col(dbc.Label("Mes"), md=3),
                    dbc.Col(
                        dcc.Slider(1, 12, 1, value=1, id="in-mes",
                                   marks={i: str(i) for i in range(1,13)}),
                        md=9
                    )
                ], className="mb-3"),

                # ------------ NDVI / EVI ------------
                dbc.Row([
                    dbc.Col(dbc.Label("NDVI"), md=6),
                    dbc.Col(dbc.Label("EVI"),  md=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Slider(ndvi_min, ndvi_max, 0.02, value=ndvi_med,
                                       id="in-ndvi", marks=ndvi_marks), md=6),
                    dbc.Col(dcc.Slider(evi_min, evi_max, 0.02, value=evi_med,
                                       id="in-evi", marks=evi_marks), md=6)
                ], className="mb-3"),

                # ------------ Precipitación ------------
                dbc.Row([
                    dbc.Col(dbc.Label("Precipitación (mm)"), md=12),
                    dbc.Col(
                        dcc.Slider(prec_min, prec_max, 5, value=prec_med,
                                   id="in-precip", marks=prec_marks),
                        md=12
                    )
                ], className="mb-3"),

                # ------------ Temperaturas ------------
                dbc.Row([
                    dbc.Col(dbc.Label("Temp. Máx. (°C)"), md=6),
                    dbc.Col(dbc.Label("Temp. Mín. (°C)"), md=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Slider(tmax_min, tmax_max, 0.5, value=tmax_med,
                                       id="in-tmax", marks=tmax_marks), md=6),
                    dbc.Col(dcc.Slider(tmin_min, tmin_max, 0.5, value=tmin_med,
                                       id="in-tmin", marks=tmin_marks), md=6)
                ], className="mb-3"),

                # ------------ Humedad ------------
                dbc.Row([
                    dbc.Col(dbc.Label("Humedad relativa (%)"), md=12),
                    dbc.Col(dcc.Slider(hum_min, hum_max, 1, value=hum_med,
                                       id="in-hum", marks=hum_marks), md=12)
                ], className="mb-3"),

                dbc.Button("Predecir rendimiento", id="btn-predict",
                           color="primary", className="w-100 mb-3"),
                html.Div(id="pred-output", className="fw-bold"),

            ])),
            md=4
        ),

        # -------- COLUMNA DERECHA: SERIE TEMPORAL --------
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Evolución mensual del rendimiento"),
                dcc.Graph(id="ts-rend", figure=fig_ts_base)
            ])),
            md=8
        )
    ], className="mb-4"),

    # ============================================================
    #          SEGUNDA FILA : Predicción vs Real + Estacionalidad
    # ============================================================
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Predicción vs historial"),
                dcc.Graph(id="pred-vs-real", figure=fig_ts_base)
            ])),
            md=7
        ),

        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Estacionalidad del rendimiento"),
                dcc.Graph(id="fig-season", figure=fig_season)
            ])),
            md=5
        )
    ], className="mb-4"),

    # ============================================================
    #          TERCERA FILA : Relación con variables
    # ============================================================
    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Relación con variables explicativas"),
                dcc.Dropdown(
                    id="dd-var",
                    options=[{"label":v,"value":v} for v in CAND_VARS],
                    value=CAND_VARS[0], clearable=False
                ),
                dcc.Graph(id="scat-rel")
            ])),
            md=12
        )
    ], className="mb-4"),

    html.Hr(),

    # ============================================================
    #       **** TABLERO FINAL DE RIESGO CLIMÁTICO ****
    # ============================================================
    html.H3("🌡️🌧️ Tablero de Riesgo Climático — Boyacá", className="mt-4"),

    dbc.Row([
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Mapa nacional (riesgo por departamento)"),
                dcc.Graph(id="mapa-riesgo")
            ])),
            md=8
        ),

        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4("Detalle del riesgo calculado"),
                html.Div(id="riesgo-texto", className="fs-5")
            ])),
            md=4
        )
    ], className="mb-4"),

    html.Br()
])

# ========================================================
# CALLBACKS DEL DASHBOARD
# ========================================================
@app.callback(
    Output("scat-rel","figure"),
    Input("dd-var","value")
)
def update_scatter(v):
    dfp=DF.dropna(subset=[v,"rendimiento"])
    fig=px.scatter(dfp,x=v,y="rendimiento",trendline="ols")
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return fig


@app.callback(
    Output("pred-output","children"),
    Output("pred-vs-real","figure"),
    Output("mapa-riesgo","figure"),
    Output("riesgo-texto","children"),
    Input("btn-predict","n_clicks"),
    State("in-ndvi","value"),
    State("in-evi","value"),
    State("in-precip","value"),
    State("in-tmax","value"),
    State("in-tmin","value"),
    State("in-hum","value"),
    State("in-mes","value"),
    prevent_initial_call=True
)
def update_pred(n, ndvi, evi, prec, tmax, tmin, hum, mes):

    req_cols = FEATURE_COLS
    row = build_pred_row(ndvi,evi,prec,tmax,tmin,hum,mes,req_cols)
    fix_imputer_dtype(PIPE)
    pred = float(PIPE.predict(row)[0])

    # -----------------------------------------
    # Serie temporal con línea roja
    # -----------------------------------------
    fig_ts = px.line(DF, x="fecha", y="rendimiento")
    fig_ts.add_hline(y=pred, line_dash="dot", line_color="red",
                     annotation_text=f"{pred:.2f} t/ha")
    fig_ts.update_layout(margin=dict(l=10,r=10,t=30,b=10))

    # -----------------------------------------
    # Riesgo multivariado
    # -----------------------------------------
    R = riesgo_multivariado(ndvi,evi,prec,tmax,tmin,hum)
    color_boy = color_riesgo(R)
    categoria = clasif_riesgo(R)

    texto_riesgo = (
    f"Riesgo total estimado: {R:.2f} "
    f"({categoria})."
)

    # -----------------------------------------
    # Mapa nacional coloreando solo Boyacá
    # -----------------------------------------
    df_map = pd.DataFrame({
    "NOMBRE_DPT": [f["properties"]["NOMBRE_DPT"] for f in GEO_COL["features"]],
})
    df_map["es_boyaca"] = df_map["NOMBRE_DPT"].str.upper() == "BOYACA"

    fig_map = go.Figure()

# ----- 1) Capa base: todo Colombia en gris -----
    fig_map.add_choropleth(
        geojson=GEO_COL,
        locations=df_map["NOMBRE_DPT"],
        z=[1.0] * len(df_map),  # valor dummy
        featureidkey="properties.NOMBRE_DPT",
        colorscale=[[0, "#e0e0e0"], [1, "#e0e0e0"]],  # gris claro
        showscale=False,
        marker_line_color="white",
        marker_line_width=0.5,
)

# ----- 2) Capa de Boyacá: coloreada según el riesgo R -----
    fig_map.add_choropleth(
        geojson=GEO_COL,
        locations=df_map.loc[df_map["es_boyaca"], "NOMBRE_DPT"],
        z=[R],  # el valor de riesgo para Boyacá
        featureidkey="properties.NOMBRE_DPT",
        colorscale=[
            [0.00, "#2ca02c"],  # muy bajo
            [0.25, "#1f77b4"],  # bajo
            [0.50, "#f2d600"],  # medio
            [0.75, "#ff7f0e"],  # alto
            [1.00, "#d62728"],  # muy alto
        ],
        zmin=0,
        zmax=1,
        colorbar_title="Riesgo",
        marker_line_color="black",
        marker_line_width=0.8,
)

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        title="Riesgo climático estimado para Boyacá",
        margin=dict(l=10, r=10, t=40, b=10)
    )



    # -----------------------------------------
    # Texto principal
    # -----------------------------------------
    q25,q75=DF["rendimiento"].quantile([0.25,0.75])
    if pred>=q75:
        msg="📈 Alta productividad esperada."
    elif pred>=q25:
        msg="⚖️ Rendimiento medio."
    else:
        msg="⚠️ Riesgo alto de pérdida."

    return (
        f"Predicción: {pred:.2f} t/ha — {msg}",
        fig_ts,
        fig_map,
        texto_riesgo
    )


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    # Railway pone el puerto en la variable de entorno PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host="0.0.0.0", port=port)