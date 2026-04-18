"""
Figuras Plotly reutilizaveis.

Todas as funcoes retornam go.Figure, sem dependencia de streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta visual consistente
COLORS = {
    "obs": "#1f2937",       # cinza-escuro
    "pred": "#2563eb",      # azul
    "pred_alt": "#dc2626",  # vermelho
    "precip": "#0ea5e9",    # azul claro
    "good": "#16a34a",      # verde
    "mid": "#f59e0b",       # laranja
    "bad": "#dc2626",       # vermelho
}

PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea",
    "#f59e0b", "#0ea5e9", "#db2777", "#737373",
]

_QUALITY_COLOR = {"Excelente": "#16a34a", "Moderada": "#f59e0b", "Limitada": "#dc2626"}


# ---------------------------------------------------------------------------
# Hidrogramas e series temporais
# ---------------------------------------------------------------------------
def plot_hydrograph(
    obs: pd.Series,
    pred: pd.Series,
    title: str = "Hidrograma observado vs predito",
    precip: pd.Series | None = None,
    pred_label: str = "Predito",
    height: int = 500,
) -> go.Figure:
    """Hidrograma com 2 eixos: precipitacao (top invertido) + vazao (embaixo)."""
    if precip is not None:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.25, 0.75],
            vertical_spacing=0.03,
        )
        fig.add_trace(go.Bar(
            x=precip.index, y=precip.values,
            marker_color=COLORS["precip"], opacity=0.7,
            name="Precipitacao",
        ), row=1, col=1)
        fig.update_yaxes(title_text="P (mm/h)", autorange="reversed", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=obs.index, y=obs.values,
            mode="lines", line=dict(color=COLORS["obs"], width=2),
            name="Observado",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred.values,
            mode="lines", line=dict(color=COLORS["pred"], width=2),
            name=pred_label,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Q (m³/s)", row=2, col=1)
        fig.update_xaxes(title_text="Tempo", row=2, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=obs.index, y=obs.values,
            mode="lines", line=dict(color=COLORS["obs"], width=2),
            name="Observado",
        ))
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred.values,
            mode="lines", line=dict(color=COLORS["pred"], width=2),
            name=pred_label,
        ))
        fig.update_yaxes(title_text="Q (m³/s)")
        fig.update_xaxes(title_text="Tempo")

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_flow_series(series: pd.Series, title: str = "Serie de vazao") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", line=dict(color=COLORS["pred"], width=1),
        name="Q (m³/s)",
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Tempo",
        yaxis_title="Q (m³/s)",
        height=400,
        hovermode="x unified",
    )
    return fig


def plot_availability_heatmap(series: pd.Series, value_col: str = "valido") -> go.Figure:
    df = series.to_frame("val")
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["valid"] = df["val"].notna().astype(int)

    pivot = df.pivot_table(values="valid", index="year", columns="month", aggfunc="mean")
    pivot.columns = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                     "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#dc2626"], [0.5, "#fbbf24"], [1, "#16a34a"]],
        colorbar=dict(title="Disponibilidade", tickformat=".0%"),
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="Disponibilidade de dados (ano × mes)",
        template="plotly_white",
        height=max(350, len(pivot) * 20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_flow_duration_curve(series: pd.Series) -> go.Figure:
    """Curva de permanencia."""
    vals = np.sort(series.dropna().values)[::-1]
    n = len(vals)
    perc = np.arange(1, n + 1) / (n + 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perc, y=vals,
        mode="lines", line=dict(color=COLORS["pred"], width=2),
    ))
    fig.update_layout(
        title="Curva de permanencia",
        template="plotly_white",
        xaxis_title="Probabilidade de excedencia (%)",
        yaxis_title="Q (m³/s)",
        yaxis_type="log",
        height=400,
    )
    fig.add_vline(x=95, line_dash="dot", line_color="gray", annotation_text="Q95")
    fig.add_vline(x=50, line_dash="dot", line_color="gray", annotation_text="Q50")
    return fig


def plot_monthly_boxplot(series: pd.Series) -> go.Figure:
    df = series.to_frame("Q")
    df["mes"] = df.index.month
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
              "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    fig = go.Figure()
    for m in range(1, 13):
        fig.add_trace(go.Box(
            y=df[df["mes"] == m]["Q"], name=months[m - 1],
            marker_color=COLORS["pred"],
            boxmean=True,
        ))
    fig.update_layout(
        title="Sazonalidade da vazao",
        template="plotly_white",
        yaxis_title="Q (m³/s)",
        yaxis_type="log",
        height=400,
        showlegend=False,
    )
    return fig


def plot_precip_timeseries(df: pd.DataFrame, title: str = "Precipitacao media areal") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["precip_mean"],
        marker_color=COLORS["precip"], opacity=0.8,
    ))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Tempo", yaxis_title="P (mm/h)",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Resultados (ablacao, contínuo)
# ---------------------------------------------------------------------------
def plot_nse_heatmap(ablation: pd.DataFrame) -> go.Figure:
    """Heatmap NSE por modelo × horizonte (a partir de summary.csv)."""
    horizons = ["NSE_1h", "NSE_3h", "NSE_6h", "NSE_12h", "NSE_24h"]
    df = ablation.set_index("Modelo")[horizons]
    df.columns = ["1h", "3h", "6h", "12h", "24h"]

    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns, y=df.index,
        colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [0.75, "#bbf7d0"], [1, "#15803d"]],
        zmin=0.5, zmax=0.85,
        colorbar=dict(title="NSE"),
        text=df.round(3).values,
        texttemplate="%{text}", textfont={"size": 11},
    ))
    fig.update_layout(
        title="Desempenho (NSE) por modelo × horizonte de previsao",
        template="plotly_white",
        xaxis_title="Horizonte de previsao",
        yaxis_title="Modelo",
        height=500,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_tradeoff(forecast: pd.DataFrame, continuous: pd.DataFrame) -> go.Figure:
    """Scatter NSE_6h (forecast) × NSE (continuous) com label dos modelos."""
    merged = forecast.merge(
        continuous.rename(columns={"Modelo": "Modelo", "NSE": "NSE_continuous"}),
        on="Modelo",
    )

    fig = go.Figure()

    for _, row in merged.iterrows():
        is_best = row["Modelo"] in {"LSTM_TTD_Base", "LSTM_TTD_Base_Fixed", "LSTM_TTD_Manning"}
        fig.add_trace(go.Scatter(
            x=[row["NSE_6h"]], y=[row["NSE_continuous"]],
            mode="markers+text",
            text=[row["Modelo"].replace("LSTM_TTD_", "").replace("_", " ")],
            textposition="top center",
            marker=dict(
                size=16 if is_best else 10,
                color=COLORS["good"] if is_best else COLORS["pred"],
                line=dict(color="black", width=1),
            ),
            name=row["Modelo"],
            showlegend=False,
        ))

    lims = [0.45, 0.90]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="gray", dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title="Trade-off: previsao 6h × simulacao continua",
        template="plotly_white",
        xaxis=dict(title="NSE — previsao 6h", range=lims),
        yaxis=dict(title="NSE — simulacao continua", range=lims),
        height=550,
    )
    return fig


def plot_nse_ranking(ablation: pd.DataFrame, horizon: str = "NSE_6h") -> go.Figure:
    df = ablation.sort_values(horizon, ascending=True)
    fig = go.Figure(go.Bar(
        x=df[horizon], y=df["Modelo"],
        orientation="h",
        marker=dict(
            color=df[horizon],
            colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#15803d"]],
            showscale=False,
        ),
        text=df[horizon].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Ranking de modelos — {horizon.replace('NSE_', 'NSE ')}",
        template="plotly_white",
        xaxis_title="NSE",
        xaxis=dict(range=[0.4, 0.9]),
        height=500,
    )
    return fig


def plot_monthly_nse(continuous_raw: list[dict], model_names: list[str]) -> go.Figure:
    """NSE mensal por modelo (linhas) para comparar estabilidade temporal."""
    fig = go.Figure()
    for i, entry in enumerate(continuous_raw):
        if entry["model_name"] not in model_names:
            continue
        months = [m["month"] for m in entry["monthly_metrics"]]
        nses = [m["nse"] for m in entry["monthly_metrics"]]
        fig.add_trace(go.Scatter(
            x=months, y=nses,
            mode="lines+markers",
            name=entry["model_name"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(
        title="NSE mensal durante simulacao continua",
        template="plotly_white",
        xaxis_title="Mes",
        yaxis_title="NSE mensal",
        yaxis=dict(range=[-5, 1]),
        height=450,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.75, line_dash="dot", line_color="green", annotation_text="Bom")
    return fig


# ---------------------------------------------------------------------------
# Ottobacias (geoespacial)
# ---------------------------------------------------------------------------
def plot_ottobacia_attr_histogram(attrs: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=attrs[col], nbinsx=30,
        marker_color=COLORS["pred"], opacity=0.85,
    ))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title=col, yaxis_title="Frequencia",
        height=350,
    )
    mean = attrs[col].mean()
    fig.add_vline(x=mean, line_dash="dash", line_color="red",
                  annotation_text=f"media = {mean:.2f}")
    return fig


def plot_ottobacia_scatter(attrs: pd.DataFrame, x: str, y: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=attrs[x], y=attrs[y],
        mode="markers",
        marker=dict(color=COLORS["pred"], size=8, opacity=0.6,
                    line=dict(color="white", width=0.5)),
        name="Ottobacias",
    ))
    # Correlacao simples
    if len(attrs) > 2:
        r = float(attrs[[x, y]].corr().iloc[0, 1])
        title = f"{y} × {x}  (r = {r:+.2f})"
    else:
        title = f"{y} × {x}"
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title=x, yaxis_title=y,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# IUH (Travel Time Distribution)
# ---------------------------------------------------------------------------
def plot_iuh(tc_values: list[float], sigma: float = 3.0, duration: float = 24.0) -> go.Figure:
    """IUH Gaussiano: h(t) = (1/(sigma*sqrt(2pi))) * exp(-(t-Tc)^2 / (2*sigma^2))."""
    t = np.linspace(0, duration, 200)
    fig = go.Figure()
    for i, tc in enumerate(tc_values):
        h = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - tc) ** 2) / (2 * sigma ** 2))
        fig.add_trace(go.Scatter(
            x=t, y=h,
            mode="lines",
            name=f"Tc = {tc:.1f} h",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(
        title=f"IUH Gaussiano (σ = {sigma:.1f} h)",
        template="plotly_white",
        xaxis_title="Tempo (h)",
        yaxis_title="h(t)",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Hiperparametros
# ---------------------------------------------------------------------------
def plot_hyperparam_scatter(df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:
    fig = px.scatter(
        df, x=x, y=y, color=color,
        size="n_params" if "n_params" in df.columns else None,
        hover_data=["model_type", "hidden_size", "num_layers", "lookback"],
        opacity=0.8,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
    )
    return fig


def plot_hyperparam_parallel(df: pd.DataFrame, target: str = "test_nse") -> go.Figure:
    """Coordenadas paralelas: hyperparams vs NSE."""
    cols_num = ["lookback", "hidden_size", "num_layers", "n_params", target]
    df_clean = df.dropna(subset=cols_num + ["model_type"]).copy()

    # Codifica categorica
    mt_codes, mt_labels = pd.factorize(df_clean["model_type"])

    dims = [
        dict(label="model_type", values=mt_codes,
             tickvals=list(range(len(mt_labels))), ticktext=list(mt_labels)),
        dict(label="lookback", values=df_clean["lookback"]),
        dict(label="hidden_size", values=df_clean["hidden_size"]),
        dict(label="num_layers", values=df_clean["num_layers"]),
        dict(label="n_params", values=df_clean["n_params"]),
        dict(label=target, values=df_clean[target]),
    ]
    fig = go.Figure(go.Parcoords(
        line=dict(color=df_clean[target], colorscale="Viridis", showscale=True,
                  cmin=df_clean[target].min(), cmax=df_clean[target].max()),
        dimensions=dims,
    ))
    fig.update_layout(
        title="Exploracao de hiperparametros (coordenadas paralelas)",
        height=550, template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# ANA (estacoes)
# ---------------------------------------------------------------------------
def plot_ana_quality_pie(catalog: pd.DataFrame) -> go.Figure:
    counts = catalog["quality_label"].value_counts().reindex(
        ["Excelente", "Moderada", "Limitada"], fill_value=0
    )
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=[_QUALITY_COLOR[l] for l in counts.index]),
        hole=0.5,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Qualidade das estacoes ANA no Brasil",
        template="plotly_white", height=400,
    )
    return fig


def plot_ana_by_state(catalog: pd.DataFrame, top_n: int = 15) -> go.Figure:
    counts = catalog.groupby("State").size().sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index,
        orientation="h", marker_color=COLORS["pred"],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} UFs por numero de estacoes fluviometricas",
        template="plotly_white",
        xaxis_title="Numero de estacoes",
        height=450,
    )
    return fig


def plot_ana_nyd_distribution(catalog: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=catalog["NYD"], nbinsx=50,
        marker_color=COLORS["pred"], opacity=0.8,
    ))
    fig.update_layout(
        title="Distribuicao da duracao das series (anos)",
        template="plotly_white",
        xaxis_title="Anos de dados (NYD)",
        yaxis_title="Numero de estacoes",
        height=400,
    )
    fig.add_vline(x=20, line_dash="dash", line_color="green",
                  annotation_text="20 anos (min recomendado)")
    return fig
