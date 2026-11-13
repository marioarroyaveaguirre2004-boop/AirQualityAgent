"""Dashboard: calidad del aire (PM2.5) en el Valle de Aburrá + agente con RAG.

Uso: streamlit run app.py
La app lee solo archivos locales (data/processed, models, data/index): arranca sin internet.
Sin GROQ_API_KEY el chat se desactiva con un aviso y el resto funciona igual.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import src.config as cfg
from src.agent import tools
from src.data.ica import pm25_to_ica

st.set_page_config(page_title="Aire Valle de Aburrá", page_icon="🌫️", layout="wide")

ICA_COLORS = {name: color for name, _, _, color in cfg.ICA_CATEGORIES}
CLASS_COLORS = {"normal": "#2e9e4f", "alerta": "#ff7e00", "critico": "#d62728"}
STATE_COLORS = ["#c6dbef", "#6baed6", "#08519c"]  # secuencial: bajo -> alto
DISCLAIMER = ("Herramienta académica (EAFIT SI3003). **No es consejo médico ni aviso oficial.** "
              "Para alertas oficiales consulta [SIATA](https://siata.gov.co) y el AMVA.")


# --- carga en caché ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, stations = tools.load_data()
    return daily.copy(), stations.copy()


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return tools.load_models()


@st.cache_resource(show_spinner="Cargando el buscador de documentos…")
def warm_retriever():
    from src.rag.retriever import get_retriever
    return get_retriever()


@st.cache_data(show_spinner=False)
def hmm_states(station_id: str) -> pd.DataFrame:
    """Estado latente filtrado P(z_t | x_1:t) más probable para cada día de la estación."""
    m = load_models()
    if "hmm" not in m:
        return pd.DataFrame(columns=["fecha", "estado"])
    daily, _ = load_data()
    s = daily[daily["estacion_id"] == station_id].set_index("fecha")["pm25"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    alpha, _ = m["hmm"].forward(np.log(s.clip(lower=1.0)).to_numpy())
    return pd.DataFrame({"fecha": s.index, "estado": alpha.argmax(axis=1), "observado": s.notna().to_numpy()})


def llm_available() -> bool:
    return bool(cfg.LLM_API_KEY) and bool(cfg.LLM_MODEL)


# --- gráficos -----------------------------------------------------------------------
def station_map(daily: pd.DataFrame, stations: pd.DataFrame, day: pd.Timestamp, selected: str) -> go.Figure:
    d = daily[daily["fecha"] == day][["estacion_id", "pm25"]]
    df = stations.merge(d, on="estacion_id", how="left")
    df["categoria"] = df["pm25"].map(lambda v: (pm25_to_ica(v) or {}).get("categoria", "Sin dato"))
    df["pm25_txt"] = df["pm25"].map(lambda v: "sin dato" if pd.isna(v) else f"{v:.1f} µg/m³")
    df["tamano"] = np.where(df["estacion_id"] == selected, 18, 11)
    cmap = {**ICA_COLORS, "Sin dato": "#9e9e9e"}
    order = [c for c in cmap if c in set(df["categoria"])]
    kwargs = dict(lat="lat", lon="lon", color="categoria", size="tamano", size_max=18,
                  color_discrete_map=cmap, category_orders={"categoria": order},
                  hover_name="nombre", hover_data={"municipio": True, "pm25_txt": True, "lat": False,
                                                   "lon": False, "tamano": False, "categoria": True},
                  labels={"pm25_txt": "PM2.5", "categoria": "Categoría ICA"}, zoom=10.2,
                  center={"lat": 6.25, "lon": -75.58}, height=460)
    # scatter_mapbox y no scatter_map: el plotly.js que trae Streamlit 1.40 no tiene el trazo
    # "scattermap" y el mapa sale vacío. carto-positron no necesita token de Mapbox.
    fig = px.scatter_mapbox(df, mapbox_style="carto-positron", **kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02))
    return fig


def time_series(daily: pd.DataFrame, station_id: str, start, end, show_hmm: bool) -> go.Figure:
    d = daily[(daily["estacion_id"] == station_id) & (daily["fecha"] >= start) & (daily["fecha"] <= end)]
    rows = 2 if show_hmm else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.82, 0.18] if show_hmm else [1.0])
    ymax = max(70.0, float(d["pm25"].max() or 0) * 1.15)
    # bandas ICA (recesivas) hasta el máximo visible
    for name, lo, hi, color in cfg.ICA_CATEGORIES:
        if lo > ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi + 1, ymax), fillcolor=color, opacity=0.12, line_width=0,
                      row=1, col=1)
    fig.add_hline(y=cfg.PM25_DAILY_LIMIT, line=dict(color="#c0392b", width=1.5, dash="dash"),
                  annotation_text="Límite diario 37 µg/m³", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Scatter(x=d["fecha"], y=d["pm25"], mode="lines+markers", name="PM2.5 diario",
                             line=dict(color="#1f3b57", width=2), marker=dict(size=4),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} µg/m³<extra></extra>"), row=1, col=1)
    if show_hmm:
        hs = hmm_states(station_id)
        hs = hs[(hs["fecha"] >= start) & (hs["fecha"] <= end)]
        names = load_models().get("hmm_extra", {}).get("estados", ["bajo", "medio", "alto"])
        meds = load_models().get("hmm_extra", {}).get("pm25_mediana_estado", [None] * 3)
        for k in range(3):
            sel = hs[hs["estado"] == k]
            fig.add_trace(go.Bar(x=sel["fecha"], y=[1] * len(sel), marker_color=STATE_COLORS[k],
                                 marker_line_width=0, width=86_400_000,
                                 name=f"Estado HMM: {names[k]} (~{meds[k]} µg/m³)",
                                 hovertemplate="%{x|%Y-%m-%d}<br>" + f"estado {names[k]}<extra></extra>"),
                          row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)
        fig.update_layout(barmode="stack", bargap=0)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", range=[0, ymax], gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.04)")
    fig.update_layout(height=430 if show_hmm else 380, margin=dict(l=10, r=10, t=10, b=10),
                      hovermode="x unified", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def probability_bars(pred: dict) -> go.Figure:
    models = [("Naive Bayes", pred["naive_bayes"]["probabilidades"])]
    if "hmm" in pred:
        models.append(("HMM", pred["hmm"]["probabilidades"]))
    fig = go.Figure()
    for cls in cfg.CLASSES:
        vals = [p[cls] for _, p in models]
        fig.add_trace(go.Bar(y=[m for m, _ in models], x=vals, orientation="h", name=cls,
                             marker_color=CLASS_COLORS[cls], marker_line=dict(color="white", width=2),
                             text=[f"{v:.0%}" if v >= 0.08 else "" for v in vals], textposition="inside",
                             hovertemplate=f"{cls}: " + "%{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack", height=170, margin=dict(l=10, r=10, t=5, b=5),
                      xaxis=dict(range=[0, 1], tickformat=".0%"), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.35))
    return fig


# --- chat ---------------------------------------------------------------------------
def render_trace(msg: dict) -> None:
    trace = msg.get("trace") or []
    tool_steps = [s for s in trace if s["tipo"] == "tool"]
    if not tool_steps and not msg.get("citations"):
        return
    with st.expander("¿Cómo llegó a esta respuesta?"):
        for s in tool_steps:
            icon = "⚠️" if s["error"] else "🔧"
            st.markdown(f"{icon} **{s['herramienta']}** `{s['argumentos']}` · {s['latencia_s']} s")
            st.caption(s["resultado"])
        if msg.get("citations"):
            st.markdown("**Citas:** " + "; ".join(f"{f}, p. {p}" for f, p in msg["citations"]))
        if msg.get("unverified"):
            st.warning("Citas que no coinciden con los fragmentos recuperados: "
                       + "; ".join(f"{f}, p. {p}" for f, p in msg["unverified"]))
        n_llm = sum(1 for s in trace if s["tipo"] == "llm")
        st.caption(f"{n_llm} llamadas al LLM · T_MAX = {cfg.T_MAX}")


def chat_panel(station_label: str, day_str: str) -> None:
    st.subheader("Pregúntale al asistente")
    if not llm_available():
        st.info("💬 Chat desactivado: falta `GROQ_API_KEY` o `LLM_MODEL` en `.env` (ver README). "
                "El resto del tablero funciona sin conexión.")
        return
    st.caption(f"Contexto enviado al agente: estación **{station_label}**, fecha **{day_str}**. "
               f"Modelo: `{cfg.LLM_MODEL}`.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_trace(msg)
    question = st.chat_input("Ej.: ¿Cómo está el aire hoy y qué recomiendan para niños?")
    if question:
        from src.agent.loop import run_agent

        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Consultando datos, modelos y documentos…"):
                warm_retriever()
                r = run_agent(question, history=history, estacion=station_label, fecha=day_str)
            st.markdown(r.answer)
            msg = {"role": "assistant", "content": r.answer, "trace": r.trace,
                   "citations": r.citations, "unverified": r.unverified_citations}
            render_trace(msg)
        st.session_state.messages.append(msg)


# --- página -------------------------------------------------------------------------
def main() -> None:
    daily, stations = load_data()
    stations = stations.sort_values(["municipio", "nombre"])
    labels = {r.estacion_id: f"{r.nombre.title()} ({r.municipio})" for r in stations.itertuples()}

    with st.sidebar:
        st.header("Selección")
        sid = st.selectbox("Estación", list(labels), format_func=labels.get,
                           index=list(labels).index("29493") if "29493" in labels else 0)
        st_days = daily.loc[daily["estacion_id"] == sid, "fecha"]
        dmin, dmax = st_days.min().date(), st_days.max().date()
        day = st.date_input("Fecha («hoy»)", value=dmax, min_value=dmin, max_value=dmax,
                            help="Los datos llegan hasta el 2025-12-31; la fecha elegida se usa como «hoy».")
        rng = st.date_input("Rango de la serie", value=(max(dmin, day - timedelta(days=180)), day),
                            min_value=dmin, max_value=dmax)
        show_hmm = st.toggle("Mostrar estado latente (HMM)", value=True)
        st.divider()
        if st.button("🧹 Limpiar chat", use_container_width=True):
            st.session_state.messages = []
        st.caption(f"LLM: {'✅ ' + cfg.LLM_MODEL if llm_available() else '❌ sin configurar'}")
        st.caption("Datos: IDEAM-SISAIRE (estaciones AMVA) + SIATA, promedios de 24 h. "
                   "Meteorología: Open-Meteo/ERA5 (modelada).")

    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (day - timedelta(days=180), day))
    day_ts, start_ts, end_ts = pd.Timestamp(day), pd.Timestamp(start), pd.Timestamp(end)
    station = stations.set_index("estacion_id").loc[sid]

    st.title("🌫️ Calidad del aire en el Valle de Aburrá")
    st.warning(DISCLAIMER, icon="⚠️")

    today = tools.get_pm25(sid, str(day))
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    if "error" in today:
        c1.metric("PM2.5 del día", "sin dato")
        c2.caption(today["error"])
    else:
        c1.metric(f"PM2.5 · {labels[sid]}", f"{today['pm25_ug_m3']} µg/m³")
        c2.markdown(f"**Categoría ICA**<br><span style='display:inline-block;width:12px;height:12px;"
                    f"border-radius:3px;background:{today['color']};border:1px solid #999'></span> "
                    f"{today['categoria_ica']} ({today['rango_categoria']})", unsafe_allow_html=True)
    with c3:
        pred = tools.predict_tomorrow(sid, str(day))
        if "error" in pred:
            st.caption(f"Predicción no disponible: {pred['error']}")
        else:
            nb = pred["naive_bayes"]
            st.markdown(f"**Predicción para {pred['fecha_prediccion']}** · más probable (NB): "
                        f"**{nb['clase_mas_probable']}**")
            st.plotly_chart(probability_bars(pred), use_container_width=True, config={"displayModeBar": False})
            perf = pred["desempeno_en_prueba"]
            st.caption(f"Desempeño en prueba 2024-2025 (NB): F1 macro {perf['f1_macro']:.2f} "
                       f"(persistencia {perf['persistencia_f1_macro']:.2f}), recall de episodios "
                       f"{perf['recall_alerta_o_critico']:.2f}, precisión {perf['precision_alerta_o_critico']:.2f}. "
                       "Clases del proyecto (normal ≤ 37, alerta 38–55, crítico ≥ 56 µg/m³): no son niveles oficiales.")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader(f"Mapa de estaciones · {day}")
        st.plotly_chart(station_map(daily, stations, day_ts, sid), use_container_width=True)
    with right:
        st.subheader("Serie de tiempo con bandas ICA")
        st.plotly_chart(time_series(daily, sid, start_ts, end_ts, show_hmm), use_container_width=True)
        if show_hmm:
            st.caption("Franja inferior: estado latente filtrado del HMM (niveles de fondo bajo/medio/alto, "
                       "no niveles oficiales).")
        with st.expander("Ver datos en tabla"):
            tbl = daily[(daily["estacion_id"] == sid) & (daily["fecha"] >= start_ts) & (daily["fecha"] <= end_ts)]
            st.dataframe(tbl[["fecha", "pm25", "categoria_ica", "clase", "lluvia_mm", "viento_medio_kmh"]]
                         .sort_values("fecha", ascending=False), hide_index=True, use_container_width=True)

    st.divider()
    chat_panel(labels[sid], str(day))


main()
