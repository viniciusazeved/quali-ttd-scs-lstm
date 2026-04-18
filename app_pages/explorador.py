"""
Explorador de hidrogramas — observado × predito por modelo e evento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_loader as dl
from plots import COLORS, plot_hydrograph

st.title("Explorador de hidrogramas")
st.caption(
    "Compare **vazão observada × vazão prevista** para qualquer um dos "
    "10 modelos e qualquer janela temporal. Útil durante a arguição: "
    "se a banca perguntar sobre um evento específico, abra aqui e "
    "mostre ao vivo."
)

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
try:
    ablation_raw = dl.load_ablation_results()
    continuous_raw = dl.load_continuous_raw()
    flow_df = dl.load_full_flow_series()
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao carregar dados: {exc}")
    st.stop()

# Modelos disponiveis
model_labels = list(dl.MODEL_LABELS.keys())

# ---------------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------------
st.sidebar.header("Controles")
mode = st.sidebar.radio("Modo", ["Ablação (previsão 24h)", "Simulação contínua"], index=0)
selected_label = st.sidebar.selectbox("Modelo", model_labels, index=3)
model_dir = dl.MODEL_LABELS[selected_label]

# ---------------------------------------------------------------------------
# Carregar predicoes
# ---------------------------------------------------------------------------
mode_key = "ablation" if mode.startswith("Ablação") else "continuous"
preds = dl.load_predictions(model_dir, mode=mode_key)

if not preds:
    st.warning(
        f"Predições de {selected_label} não encontradas para este modo. "
        "Tente outro modelo ou troque o modo."
    )
    st.stop()

obs = preds.get("observed")
pred = preds.get("predicted")
timestamps = preds.get("timestamps")

if obs is None or pred is None:
    st.error("Formato inesperado do .npz. Chaves disponíveis: " +
             ", ".join(preds.keys()))
    st.stop()

# Formato da ablacao: obs (N, 24) e pred (N, 24) — matrizes multi-horizonte
# Formato da continua: obs (T,) e pred (T,) — vetores ao longo do tempo
obs = np.asarray(obs)
pred = np.asarray(pred)

st.sidebar.divider()
if obs.ndim == 2:
    horizon_options = [f"+{h}h" for h in [1, 3, 6, 12, 24] if h - 1 < obs.shape[1]]
    selected_h = st.sidebar.selectbox("Horizonte de previsão", horizon_options, index=2)
    h_idx = int(selected_h.replace("+", "").replace("h", "")) - 1
    obs_series = obs[:, h_idx]
    pred_series = pred[:, h_idx]
    st.sidebar.caption(f"Shape: {obs.shape} (n_amostras, n_horizontes)")
else:
    obs_series = obs
    pred_series = pred
    selected_h = "—"

# Construir index temporal
if timestamps is not None and len(timestamps) == len(obs_series):
    ts = pd.DatetimeIndex(timestamps)
else:
    # Ablation nao grava timestamps. Alinha o fim dos samples com test_end
    # (aproximacao boa o bastante para visualizacao).
    info = dl.load_dataset_info()
    test_end = pd.Timestamp(info["splits"]["test"]["end"])
    ts = pd.date_range(end=test_end, periods=len(obs_series), freq="h")

obs_s = pd.Series(obs_series, index=ts, name="observado")
pred_s = pd.Series(pred_series, index=ts, name="predito")

# ---------------------------------------------------------------------------
# KPIs de desempenho (sobre a janela completa)
# ---------------------------------------------------------------------------
valid = (~obs_s.isna()) & (~pred_s.isna())
o = obs_s[valid].values
p = pred_s[valid].values

if len(o) > 10:
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    pbias = float(100 * (p.sum() - o.sum()) / o.sum())
    r2 = float(np.corrcoef(o, p)[0, 1]) ** 2
else:
    nse = rmse = pbias = r2 = float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("NSE", f"{nse:.3f}")
k2.metric("RMSE (m³/s)", f"{rmse:.2f}")
k3.metric("PBIAS (%)", f"{pbias:+.2f}")
k4.metric("R²", f"{r2:.3f}")

# ---------------------------------------------------------------------------
# Seletor temporal
# ---------------------------------------------------------------------------
st.divider()

min_date = ts.min().to_pydatetime() if hasattr(ts, 'min') else None
max_date = ts.max().to_pydatetime() if hasattr(ts, 'max') else None

col_a, col_b = st.columns([0.7, 0.3])
with col_a:
    if min_date and max_date:
        date_range = st.slider(
            "Janela temporal",
            min_value=min_date, max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
        mask = (ts >= date_range[0]) & (ts <= date_range[1])
        obs_w = obs_s[mask]
        pred_w = pred_s[mask]
    else:
        obs_w = obs_s
        pred_w = pred_s

with col_b:
    show_precip = st.checkbox("Exibir precipitação", True)

# Precipitacao correspondente (media areal do HDF5)
if show_precip:
    precip_series = flow_df["precip_mean"].reindex(obs_w.index)
else:
    precip_series = None

# ---------------------------------------------------------------------------
# Hidrograma
# ---------------------------------------------------------------------------
title = (
    f"{selected_label} — {mode}"
    + (f" — horizonte {selected_h}" if obs.ndim == 2 else "")
)

st.plotly_chart(
    plot_hydrograph(obs_w, pred_w, title=title,
                    precip=precip_series, pred_label=selected_label),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Scatter obs vs pred
# ---------------------------------------------------------------------------
st.subheader("Scatter observado × predito")
col_s1, col_s2 = st.columns(2)

with col_s1:
    # Scatter
    o_w = obs_w.dropna().values
    p_w = pred_w.reindex(obs_w.dropna().index).values
    lim = max(o_w.max(), p_w[~np.isnan(p_w)].max()) * 1.05

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=o_w, y=p_w,
        mode="markers",
        marker=dict(color=COLORS["pred"], size=5, opacity=0.4),
        name="Pares",
    ))
    fig_sc.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines",
        line=dict(color="red", dash="dash"), name="1:1",
    ))
    fig_sc.update_layout(
        title="Observado × Predito",
        template="plotly_white",
        xaxis_title="Observado (m³/s)",
        yaxis_title="Predito (m³/s)",
        height=450,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_s2:
    # Residuos temporais
    res = (pred_w - obs_w).dropna()
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=res.index, y=res.values,
        mode="lines", line=dict(color=COLORS["bad"], width=1),
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_res.update_layout(
        title="Resíduos (predito − observado)",
        template="plotly_white",
        xaxis_title="Tempo",
        yaxis_title="Resíduo (m³/s)",
        height=450,
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ---------------------------------------------------------------------------
# Detecção automática de eventos de cheia
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Eventos de cheia (pico > Q95)")

q95 = float(np.nanpercentile(obs_w.values, 95))

# Eventos são picos onde obs > q95, com separação de 48h entre picos
obs_arr = obs_w.values.copy()
obs_arr[np.isnan(obs_arr)] = -np.inf
peaks = []
i = 1
while i < len(obs_arr) - 1:
    if obs_arr[i] > q95 and obs_arr[i] > obs_arr[i - 1] and obs_arr[i] > obs_arr[i + 1]:
        peaks.append(i)
        i += 48  # separacao entre picos
    else:
        i += 1

st.caption(f"Q95 = {q95:.1f} m³/s · {len(peaks)} eventos detectados na janela")

if peaks:
    event_options = [
        f"Evento {i + 1}: {obs_w.index[p]:%Y-%m-%d %H:%M} (Q_pico = {obs_arr[p]:.1f} m³/s)"
        for i, p in enumerate(peaks)
    ]
    evt = st.selectbox("Selecione um evento", event_options)
    idx = event_options.index(evt)
    peak_pos = peaks[idx]

    # Janela: 72h antes, 120h depois
    i0 = max(0, peak_pos - 72)
    i1 = min(len(obs_w), peak_pos + 120)
    obs_e = obs_w.iloc[i0:i1]
    pred_e = pred_w.iloc[i0:i1]
    if show_precip and precip_series is not None:
        precip_e = precip_series.iloc[i0:i1]
    else:
        precip_e = None

    st.plotly_chart(
        plot_hydrograph(
            obs_e, pred_e,
            title=f"Evento em {obs_w.index[peak_pos]:%Y-%m-%d}",
            precip=precip_e,
            pred_label=selected_label,
            height=450,
        ),
        use_container_width=True,
    )

    # Metricas do evento
    o = obs_e.dropna().values
    p = pred_e.reindex(obs_e.dropna().index).values
    mask_e = ~np.isnan(p)
    o = o[mask_e]
    p = p[mask_e]
    if len(o) > 10:
        nse_e = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
        pbias_e = 100 * (p.sum() - o.sum()) / o.sum()
        peak_obs = o.max()
        peak_pred = p.max()
        err_peak = 100 * (peak_pred - peak_obs) / peak_obs

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NSE (evento)", f"{nse_e:.3f}")
        c2.metric("PBIAS (%)", f"{pbias_e:+.2f}")
        c3.metric("Pico observado (m³/s)", f"{peak_obs:.1f}")
        c4.metric("Erro do pico (%)", f"{err_peak:+.1f}")
