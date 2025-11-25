# app_dash_boyaca_v2.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.impute import SimpleImputer

# ===================== Carga de artefactos =====================
BASE_DIR = Path(__file__).resolve().parent
ARTI = BASE_DIR / "artifacts"

DF       = pd.read_csv(ARTI / "dataset_modelo.csv")
DF_RES   = pd.read_csv(ARTI / "df_resultados.csv")
META     = json.loads((ARTI / "metadata.json").read_text(encoding="utf-8"))
PIPE     = joblib.load(ARTI / "modelo_boyaca.pkl")
FEATURE_COLS = joblib.load(ARTI / "feature_cols.pkl")

# ---------- Parche robusto para SimpleImputer ----------
def fix_imputer_dtype(est):
    """Asegura que cualquier SimpleImputer tenga _fit_dtype válido."""
    def _touch(x):
        if isinstance(x, SimpleImputer):
            kd = getattr(getattr(x, "_fit_dtype", None), "kind", None)
            if kd is None:
                x._fit_dtype = np.dtype("float64")
    _touch(est)
    if hasattr(est, "named_steps"):
        for s in est.named_steps.values():
            fix_imputer_dtype(s)

fix_imputer_dtype(PIPE)

# ===================== Normalización mínima =====================
if "fecha" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["fecha"], errors="coerce")
elif "yyyymm" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["yyyymm"], format="%Y-%m", errors="coerce")

DF["rendimiento"] = pd.to_numeric(DF["rendimiento"], errors="coerce")
DF = DF.dropna(subset=["fecha", "rendimiento"]).copy()
DF = DF.sort_values("fecha")

# columnas candidatas para análisis
CAND_VARS = [c for c in [
    "NDVI", "EVI", "Precipitacion", "TempMax", "TempMin", "HumedadRelativa"
] if c in DF.columns]

# columnas temporales para análisis estacional
if "mes" in DF.columns:
    DF["mes_num"] = DF["mes"].astype(int)
else:
    DF["mes_num"] = DF["fecha"].dt.month

DF["mes_nombre"] = DF["mes_num"].map({
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep",10: "Oct",11: "Nov",12: "Dic"
})
DF["anio"] = DF["fecha"].dt.year

DF_SORTED = DF.sort_values("fecha").copy()

# ===================== Utilidades del pipeline =====================

def _iter_steps(est):
    yield est
    if hasattr(est, "named_steps"):
        for s in est.named_steps.values():
            yield from _iter_steps(s)

def required_cols_from_pipe(pipe, fallback):
    """Intenta recuperar feature_names_in_ desde cualquier step."""
    for est in _iter_steps(pipe):
        cols = getattr(est, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    return list(fallback)

ALIASES = {
    "ndvi": ["ndvi"],
    "evi": ["evi"],
    "precipitacion": ["prec", "precipitacion", "rain", "ppt", "mm"],
    "tempmax": ["tmax", "tempmax", "temp_max", "max"],
    "tempmin": ["tmin", "tempmin", "temp_min", "min"],
    "humedadrelativa": ["hum", "humedad", "rh", "humedadrelativa"],
    "month_sin": ["month_sin", "mes_sin"],
    "month_cos": ["month_cos", "mes_cos"],
    "mes": ["mes", "month"],
    "anio": ["anio", "year"],
    "fecha": ["fecha", "date", "yyyymm", "yearmonth"]
}

def build_pred_row(ndvi, evi, precip, tmax, tmin, hum, mes, req_cols):
    """Construye una fila de predicción con el mismo orden de columnas del pipeline."""
    m_sin = np.sin(2*np.pi*mes/12.0)
    m_cos = np.cos(2*np.pi*mes/12.0)

    values = {
        "ndvi": float(ndvi),
        "evi": float(evi),
        "precipitacion": float(precip),
        "tempmax": float(tmax),
        "tempmin": float(tmin),
        "humedadrelativa": float(hum),
        "month_sin": m_sin,
        "month_cos": m_cos,
        "mes": int(mes),
        "anio": int(DF["anio"].max()),  # último año del dataset
        "fecha": int(DF["anio"].max()*100 + mes),
    }

    def match_value(col):
        cl = col.lower()
        if cl in values:
            return values[cl]
        for key, al in ALIASES.items():
            if any(a in cl for a in al) and key in values:
                return values[key]
        return np.nan

    row = pd.DataFrame([{c: match_value(c) for c in req_cols}], columns=req_cols)
    return row.apply(pd.to_numeric, errors="coerce")

# ===================== Componentes visuales =====================

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

# controles de predicción (se reusan en sidebar)
controls_pred = dbc.Card(
    dbc.CardBody([
        html.H5("Predicción interactiva", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Label("Mes"), width=3),
            dbc.Col(
                dcc.Slider(
                    1, 12, 1, value=1, id="in-mes",
                    marks={i: str(i) for i in range(1, 13)}
                ),
                width=9
            )
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Label("NDVI"), md=6),
            dbc.Col(dbc.Label("EVI"),  md=6)
        ], className="mt-2"),
        dbc.Row([
            dbc.Col(
                dcc.Slider(
                    0.2, 0.95, 0.01,
                    value=float(DF["NDVI"].median()) if "NDVI" in DF else 0.6,
                    id="in-ndvi"
                ),
                md=6
            ),
            dbc.Col(
                dcc.Slider(
                    0.1, 0.9, 0.01,
                    value=float(DF["EVI"].median()) if "EVI" in DF else 0.4,
                    id="in-evi"
                ),
                md=6
            ),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Label("Precipitación (mm)"), md=12),
            dbc.Col(
                dcc.Slider(
                    60, 300, 1,
                    value=float(DF["Precipitacion"].median())
                    if "Precipitacion" in DF else 160,
                    id="in-precip"
                ),
                md=12
            )
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Temp. Máx. (°C)"), md=6),
            dbc.Col(dbc.Label("Temp. Mín. (°C)"), md=6)
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Slider(
                    20, 30, 0.1,
                    value=float(DF["TempMax"].median()) if "TempMax" in DF else 25,
                    id="in-tmax"
                ),
                md=6
            ),
            dbc.Col(
                dcc.Slider(
                    10, 20, 0.1,
                    value=float(DF["TempMin"].median()) if "TempMin" in DF else 15,
                    id="in-tmin"
                ),
                md=6
            ),
        ]),
        dbc.Row([
            dbc.Col(dbc.Label("Humedad Relativa (%)"), md=12),
            dbc.Col(
                dcc.Slider(
                    60, 90, 1,
                    value=float(DF["HumedadRelativa"].median())
                    if "HumedadRelativa" in DF else 75,
                    id="in-hum"
                ),
                md=12
            )
        ], className="mb-3"),

        dbc.Button("Predecir rendimiento", id="btn-predict",
                   color="primary", className="w-100"),
        html.Div(id="pred-output", className="mt-3 fw-bold"),
    ]),
    className="shadow-sm",
    style={"backgroundColor": "#fafafa"}
)

# ---------- Sidebar con filtros tipo mockup ----------
sidebar = dbc.Card(
    dbc.CardBody([
        html.H4("Panel Boyacá", className="mb-4 text-center"),
        dbc.Label("Nivel de riesgo"),
        dcc.Dropdown(
            id="dd-nivel",
            options=[
                {"label": "Bajo", "value": "Bajo"},
                {"label": "Medio", "value": "Medio"},
                {"label": "Alto", "value": "Alto"},
            ],
            value="Medio",
            clearable=False,
            className="mb-3"
        ),
        dbc.Label("Ubicación"),
        dcc.Dropdown(
            id="dd-ubicacion",
            options=[{"label": "Boyacá - Departamento", "value": "Boyacá"}],
            value="Boyacá",
            clearable=False,
            className="mb-3"
        ),
        dbc.Label("Municipio"),
        dcc.Dropdown(
            id="dd-muni",
            options=[{"label": "Zona cafetera Boyacá (promedio)", "value": "Zona cafetera"}],
            value="Zona cafetera",
            clearable=False,
            className="mb-4"
        ),

        html.Hr(),
        controls_pred
    ]),
    className="shadow-sm",
    style={"height": "100%", "backgroundColor": "#222", "color": "#f8f9fa"}
)

# ===================== Figuras base =====================

# KPIs (usamos claves flexibles por si cambia metadata)
metricas = META.get("metricas_test_snapshot") or META.get("metricas_notebook") or {}
R2_val   = metricas.get("R2", 0.0)
RMSE_val = metricas.get("RMSE", 0.0)
MAE_val  = metricas.get("MAE", 0.0)
N_obs    = META.get("n_obs", len(DF))

# serie de tiempo base
fig_ts = px.line(
    DF_SORTED, x="fecha", y="rendimiento",
    labels={"fecha": "Fecha", "rendimiento": "t/ha"}
).update_layout(margin=dict(l=10, r=10, t=10, b=10))

# figura de estacionalidad (caja por mes)
fig_season = px.box(
    DF, x="mes_nombre", y="rendimiento",
    category_orders={"mes_nombre": ["Ene","Feb","Mar","Abr","May","Jun",
                                    "Jul","Ago","Sep","Oct","Nov","Dic"]},
    labels={"mes_nombre": "Mes", "rendimiento": "t/ha"},
    title="Distribución mensual del rendimiento"
).update_layout(margin=dict(l=10, r=10, t=40, b=10))

# comparación de modelos (barras)
fig_modelos = px.bar(
    DF_RES, x="Modelo", y=["R2", "RMSE", "MAE"],
    barmode="group",
    title="Comparación de métricas entre modelos"
).update_layout(margin=dict(l=10, r=10, t=40, b=10))

# mapa inicial (Boyacá centro aproximado)
def make_mapa(nivel="Medio"):
    riesgo_map = {"Bajo": 0.25, "Medio": 0.5, "Alto": 0.85}
    val = riesgo_map.get(nivel, 0.5)

    df_map = pd.DataFrame({
        "lugar": ["Boyacá (promedio)"],
        "lat": [5.545],   # coordenadas aproximadas del departamento
        "lon": [-73.362],
        "riesgo": [val]
    })

    fig = px.scatter_geo(
        df_map, lat="lat", lon="lon", color="riesgo",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
        range_color=(0, 1),
        size=[20],
        hover_name="lugar",
        projection="natural earth",
        title="Mapa de riesgo – Departamento de Boyacá"
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        geo=dict(
            scope="south america",
            showcountries=True,
            countrycolor="LightGray",
            lataxis_range=[-5, 15],
            lonaxis_range=[-80, -65]
        )
    )
    return fig

fig_mapa_init = make_mapa("Medio")

# ===================== App y layout =====================

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Dashboard Boyacá")

main_content = dbc.Container(fluid=True, children=[
    html.Br(),
    dbc.Row([
        dbc.Col(html.H2("📊 Dashboard – Rendimiento de Café (Boyacá)"), md=8),
        dbc.Col(html.Div(f"Modelo: {META.get('model', 'RandomForestRegressor')}"),
                md=4, className="text-end align-self-center")
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(kpi_card("R² (test)",   f"{R2_val:.3f}",   color="success"), md=3),
        dbc.Col(kpi_card("RMSE (test)", f"{RMSE_val:.3f}", color="danger"),  md=3),
        dbc.Col(kpi_card("MAE (test)",  f"{MAE_val:.3f}",  color="warning"), md=3),
        dbc.Col(kpi_card("Observaciones", f"{N_obs:,}"), md=3),
    ], className="g-3 mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Predicción", tab_id="tab-pred", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Evolución mensual del rendimiento"),
                    dcc.Graph(id="ts-rend", figure=fig_ts)
                ])), md=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Predicción vs historial"),
                    dcc.Graph(id="pred-vs-real", figure=fig_ts),
                    html.Div(
                        "La línea punteada indica el rendimiento estimado "
                        "para la combinación de clima y vegetación seleccionada.",
                        className="text-muted small mt-2"
                    )
                ])), md=12)
            ])
        ]),

        dbc.Tab(label="Análisis histórico", tab_id="tab-analisis", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Relación con variables"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id="dd-var",
                            options=[{"label": v, "value": v} for v in CAND_VARS]
                            or [{"label": "NDVI", "value": "NDVI"}],
                            value=CAND_VARS[0] if CAND_VARS else "NDVI",
                            clearable=False
                        ), md=4)
                    ], className="mb-2"),
                    dcc.Graph(id="scat-rel")
                ])), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Estacionalidad del rendimiento"),
                    dcc.Graph(id="fig-season", figure=fig_season)
                ])), md=5),
            ])
        ]),

        dbc.Tab(label="Modelos", tab_id="tab-modelos", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Tabla de resultados de modelos"),
                    dash_table.DataTable(
                        id="tbl-result",
                        data=DF_RES.round(4).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in DF_RES.columns],
                        sort_action="native",
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "Inter, system-ui",
                            "fontSize": "14px",
                            "padding": "6px"
                        },
                        style_header={"fontWeight": "700"}
                    )
                ])), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Comparación gráfica de métricas"),
                    dcc.Graph(id="fig-modelos", figure=fig_modelos)
                ])), md=6),
            ])
        ]),

        dbc.Tab(label="Mapa Boyacá", tab_id="tab-mapa", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Mapa de riesgo para Boyacá"),
                    dcc.Graph(id="fig-mapa", figure=fig_mapa_init),
                    html.Div(id="texto-mapa", className="mt-2")
                ])), md=12)
            ])
        ]),
    ])
])

app.layout = dbc.Container(fluid=True, children=[
    html.Br(),
    dbc.Row([
        dbc.Col(sidebar, md=3, lg=3),
        dbc.Col(main_content, md=9, lg=9)
    ])
])

# ===================== Callbacks =====================

# scatter de relación con variables
@app.callback(
    Output("scat-rel", "figure"),
    Input("dd-var", "value")
)
def update_scatter(var):
    dfp = DF.dropna(subset=[var, "rendimiento"])
    fig = px.scatter(
        dfp, x=var, y="rendimiento", trendline="ols",
        labels={var: var, "rendimiento": "t/ha"},
        title=f"Rendimiento vs {var}"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

# predicción + figura pred vs historial
@app.callback(
    Output("pred-output", "children"),
    Output("pred-vs-real", "figure"),
    Input("btn-predict", "n_clicks"),
    State("in-ndvi", "value"),
    State("in-evi", "value"),
    State("in-precip", "value"),
    State("in-tmax", "value"),
    State("in-tmin", "value"),
    State("in-hum", "value"),
    State("in-mes", "value"),
    prevent_initial_call=True
)
def predict_click(nc, ndvi, evi, precip, tmax, tmin, hum, mes):
    try:
        req_cols = required_cols_from_pipe(PIPE, FEATURE_COLS)
        fila = build_pred_row(ndvi, evi, precip, tmax, tmin, hum, mes, req_cols)
        fix_imputer_dtype(PIPE)
        pred = float(PIPE.predict(fila)[0])

        # figura: serie + línea horizontal con la predicción
        fig = px.line(
            DF_SORTED, x="fecha", y="rendimiento",
            labels={"fecha": "Fecha", "rendimiento": "t/ha"}
        )
        fig.add_hline(
            y=pred,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Predicción {pred:.2f} t/ha"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        # interpretación simple de riesgo
        q25, q75 = DF["rendimiento"].quantile([0.25, 0.75])
        if pred >= q75:
            texto_riesgo = "📈 Alta productividad esperada. Riesgo de pérdida bajo."
        elif pred >= q25:
            texto_riesgo = "⚖️ Rendimiento dentro de un rango medio/histórico."
        else:
            texto_riesgo = "⚠️ Rendimiento bajo. Riesgo elevado, el seguro sería relevante."

        return f"✅ Rendimiento estimado: {pred:.2f} t/ha. {texto_riesgo}", fig

    except Exception as e:
        # si algo falla, devolvemos la serie base
        return f"⚠️ Error al predecir: {e}", fig_ts

# mapa de riesgo según el nivel seleccionado
@app.callback(
    Output("fig-mapa", "figure"),
    Output("texto-mapa", "children"),
    Input("dd-nivel", "value"),
    Input("dd-ubicacion", "value"),
    Input("dd-muni", "value")
)
def update_mapa(nivel, ubicacion, muni):
    if nivel is None:
        nivel = "Medio"
    fig = make_mapa(nivel)

    mensajes = {
        "Bajo":  "Nivel de riesgo bajo: condiciones cercanas a las mejores observadas.",
        "Medio": "Nivel de riesgo medio: variabilidad moderada, el seguro puede apoyar estabilidad.",
        "Alto":  "Nivel de riesgo alto: probabilidad importante de pérdidas, el seguro cobra mayor relevancia."
    }
    texto = f"Zona: {ubicacion} – {muni}. {mensajes.get(nivel, '')}"
    return fig, texto

# ===================== Main =====================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
