"""
Pagina de dados — serie de vazao, precipitacao, disponibilidade, sazonalidade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_loader as dl
from plots import (
    COLORS,
    plot_availability_heatmap,
    plot_flow_duration_curve,
    plot_flow_series,
    plot_monthly_boxplot,
    plot_precip_timeseries,
)
from embed_mode import maybe_hide_chrome

maybe_hide_chrome()

st.title("Dados de entrada")
st.caption(
    "Quais dados o modelo vê: precipitação de satélite (MERGE/CPTEC, "
    "horária), vazão observada para avaliação (ANA 58585000 — Manuel "
    "Duarte) e atributos fisiográficos da bacia"
)

with st.expander("Fontes de dados — detalhes técnicos"):
    st.markdown(
        """
        **Vazão — estação ANA 58585000 (Manuel Duarte)**
        Telemétrica, resolução horária. Série 2013–2025 disponível, mas
        **2021–2025 foi selecionado** por apresentar a melhor combinação de
        continuidade e disponibilidade (média de 91,6% de dados válidos).
        Anos anteriores foram excluídos: 2014 (48%), 2015 (68%), 2020 (41%).

        **Precipitação — MERGE (CPTEC/INPE)** *(Rozante et al., 2010)*
        Combina estimativas de satélite (TRMM/GPM) com observações de
        superfície (ANA, DAEE, SIMEPAR, CEMADEN, INEMA) via interpolação
        ótima. Resolução espacial 0,1° (~10 km, ~123 km²/pixel), horária,
        cobertura América do Sul desde jun/2000.
        **Justificativa da agregação espacial:** pixel MERGE (~123 km²)
        é muito maior que a ottobacia média (12,7 km²) — a precipitação é
        atribuída às 245 ottobacias por **intersecção ponderada pela área
        de sobreposição**.

        **Modelo Digital de Terreno — ANADEM v1** *(Laipelt et al., 2024)*
        IPH/UFRGS + ANA (TED nº 05/2019). Derivado do Copernicus GLO-30
        com **correção de viés de vegetação** por aprendizado de máquina
        (GEDI + Landsat-8 + Sentinel-2). Validação ICESat-2: **redução de
        85% no viés** vs Copernicus original. Resolução 30 m, datum
        WGS1984/EGM2008. Asset GEE: `projects/et-brasil/assets/anadem/v1`.

        **Curve Number — BHAE_CN-2022 (ANA)** *(Nota Técnica nº 9/2025/COMUC/SHE)*
        Escala 1:250.000. Combina solos IBGE (2023) com MapBiomas Col. 8.0
        (ano-base 2022). Usa a **classificação hidrológica de Sartori (2005)**
        adaptada para solos tropicais brasileiros (Latossolos, Argissolos).

        **Uso e cobertura do solo — MapBiomas Coleção 8.0**
        Ano-base 2022, resolução 30 m (Landsat) / 10 m (Sentinel-2 desde 2019).
        As 30+ classes originais foram reagrupadas em 7 classes por
        similaridade de rugosidade de Manning (floresta, veg. natural,
        agropecuária, solo exposto, água, urbano, não observado).
        """
    )

try:
    info = dl.load_dataset_info()
    flow_df = dl.load_full_flow_series()
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao carregar dataset: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Passos temporais", f"{int(sum(s['n_steps'] for s in info['splits'].values())):,}".replace(",", "."))
k2.metric("Vazão média", f"{flow_df['streamflow'].mean():.1f} m³/s")
k3.metric("Vazão máxima", f"{flow_df['streamflow'].max():.1f} m³/s")
k4.metric("Ottobacias", f"{info['n_ottobacias']}")

# ---------------------------------------------------------------------------
# Splits temporais
# ---------------------------------------------------------------------------
st.subheader("Divisão temporal (train / val / test)")

split_rows = []
for name, stats in info["splits"].items():
    split_rows.append({
        "Split": name,
        "Início": stats["start"],
        "Fim": stats["end"],
        "Passos (h)": stats["n_steps"],
        "Anos": round(stats["n_steps"] / 8760, 2),
        "Q média (m³/s)": round(stats["mean_q"], 2),
        "Q máxima (m³/s)": round(stats["max_q"], 1),
    })
split_df = pd.DataFrame(split_rows)
st.dataframe(
    split_df.style.format({
        "Início": "{:%Y-%m-%d}", "Fim": "{:%Y-%m-%d}",
        "Passos (h)": "{:,.0f}",
    }),
    use_container_width=True, hide_index=True,
)

# Visualizacao dos splits
st.markdown("Visualização dos splits sobre a série de vazão:")
fig_splits = go.Figure()
colors = {"train": "#93c5fd", "val": "#fde68a", "test": "#fca5a5"}
for split in ("train", "val", "test"):
    mask = flow_df["split"] == split
    if mask.any():
        fig_splits.add_trace(go.Scatter(
            x=flow_df.index[mask], y=flow_df.loc[mask, "streamflow"],
            mode="lines", line=dict(width=1, color=colors[split]),
            name=split,
        ))
fig_splits.update_layout(
    template="plotly_white",
    xaxis_title="Tempo", yaxis_title="Q (m³/s)",
    height=400, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_splits, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs analiticas
# ---------------------------------------------------------------------------
st.divider()
tab_serie, tab_precip, tab_perm, tab_saz, tab_disp = st.tabs([
    "Série de vazão", "Precipitação", "Curva de permanência",
    "Sazonalidade", "Disponibilidade",
])

with tab_serie:
    st.plotly_chart(
        plot_flow_series(flow_df["streamflow"], "Vazão horária — Manuel Duarte"),
        use_container_width=True,
    )
    st.caption(
        "Série horária concatenada (train + val + test) na estação 58585000. "
        "Os valores são derivados da telemetria ANA com tratamento para picos "
        "espúrios e lacunas."
    )

with tab_precip:
    st.plotly_chart(
        plot_precip_timeseries(flow_df, "Precipitação média areal (MERGE/CPTEC)"),
        use_container_width=True,
    )

    # Totais anuais (YE = year-end; ms/h * h = mm)
    annual_p = flow_df["precip_mean"].resample("YE").sum()
    fig_annual = go.Figure(go.Bar(
        x=annual_p.index.year, y=annual_p.values,
        marker_color=COLORS["precip"], opacity=0.8,
    ))
    fig_annual.add_hline(y=annual_p.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"média = {annual_p.mean():.0f} mm/ano")
    fig_annual.update_layout(
        title="Precipitação anual (mm)",
        template="plotly_white",
        xaxis_title="Ano", yaxis_title="P (mm)",
        height=350,
    )
    st.plotly_chart(fig_annual, use_container_width=True)

    st.caption(
        "A precipitação horária é o produto MERGE do CPTEC/INPE (~10 km), "
        "que combina estimativas de satélite (TRMM/GPM) com observações "
        "de superfície. Cobertura desde junho/2000. A série mostrada é a "
        "média espacial sobre as 245 ottobacias."
    )

with tab_perm:
    st.plotly_chart(
        plot_flow_duration_curve(flow_df["streamflow"]),
        use_container_width=True,
    )
    q50 = flow_df["streamflow"].quantile(0.5)
    q95 = flow_df["streamflow"].quantile(0.05)  # Q95 = prob. excedencia 95%
    st.caption(
        f"Q50 = {q50:.1f} m³/s · Q95 = {q95:.1f} m³/s · "
        f"Q5 (cheia) = {flow_df['streamflow'].quantile(0.95):.1f} m³/s"
    )

with tab_saz:
    st.plotly_chart(
        plot_monthly_boxplot(flow_df["streamflow"]),
        use_container_width=True,
    )
    st.caption(
        "A bacia apresenta regime sazonal marcado (estações chuvosa e seca). "
        "Eixo Y em escala logarítmica para permitir visualização conjunta "
        "de eventos de cheia e de estiagem."
    )

with tab_disp:
    st.plotly_chart(
        plot_availability_heatmap(flow_df["streamflow"]),
        use_container_width=True,
    )
    st.caption(
        "Disponibilidade da série de vazão. Valores próximos de 100% indicam "
        "meses com dados completos; valores inferiores indicam lacunas ou "
        "dados espúrios removidos no pré-processamento."
    )
