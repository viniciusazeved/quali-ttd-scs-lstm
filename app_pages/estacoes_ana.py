"""
Pagina de contexto — estacoes fluviometricas CONVENCIONAIS da ANA no Brasil.

Convencional = leitura diaria em regua/linigrafo, registro manual.
Serve como ponto de partida da narrativa: apesar da rede densa, a operacao
convencional fornece apenas vazao DIARIA, incompativel com modelagem de
eventos de cheia e previsao horaria.
"""
from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

import stations as ana
from plots import plot_ana_by_state, plot_ana_nyd_distribution, plot_ana_quality_pie

# Data de referência para "estações recentes" = último ano presente no catálogo.
# O ANAF foi compilado em março/2020; usar datetime.now() daria zero.
CATALOG_END_YEAR = 2020
RECENT_YEARS = 5  # janela: últimos 5 anos do catálogo

st.title("Monitoramento convencional da ANA")
st.caption(
    "Rede fluviometrica de leitura diaria — catalogo ANAF (Carvalho & Braga, 2020)"
)

st.markdown(
    """
    A **ANA** coordena a Rede Hidrometeorologica Nacional (RHN), hoje com
    cerca de **4.500 estacoes fluviometricas** distribuidas pelo Brasil.
    A maioria opera em **regime convencional**: leitura diaria em regua ou
    linigrafo, registro manual por observador local, transmissao em lote.

    Esta pagina mapeia o **catalogo ANAF** — subconjunto das estacoes
    fluviometricas com dados efetivamente registrados e disponiveis via
    HidroWeb. O passo seguinte da narrativa mostra a **subrede telemetrica**,
    aquela que fornece o dado **horario** exigido pelo modelo TTD-SCS-LSTM.
    """
)

st.info(
    "📌 **Convencional = vazao diaria.** O modelo TTD-SCS-LSTM opera em "
    "passo horario, entao estas estacoes *nao* servem como exutorio para "
    "treinamento/avaliacao. Servem como contexto da rede total e como "
    "fonte de longas series historicas (decadas) para estudos de regime."
)

# ---------------------------------------------------------------------------
# Toggle: todas vs apenas ANA (RHN pura)
# ---------------------------------------------------------------------------
recorte = st.radio(
    "Recorte da rede",
    options=["Todas as operadoras", "Apenas ANA (gestao federal)"],
    horizontal=True,
    help=(
        "A coluna Responsible do ANAF indica quem responde pela estacao. "
        "'Apenas ANA' filtra as 2.694 estacoes sob gestao federal direta; "
        "'Todas' inclui DAEE-SP, COPEL, FURNAS, CPRM-SGB, AGUASPARANA etc."
    ),
)
only_ana = recorte.startswith("Apenas")

try:
    catalog = ana.load_convencionais(only_ana=only_ana)
except Exception as exc:  # noqa: BLE001
    st.error(f"Nao foi possivel carregar o catalogo convencional: {exc}")
    st.stop()

scored = ana.compute_quality_score(catalog) if "quality_score" not in catalog.columns else catalog

# ---------------------------------------------------------------------------
# Filtros (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros do mapa")
min_years = st.sidebar.slider("Anos mínimos de dados", 0, 80, 10, 5)

all_states = ana.get_states(scored)
ufs_sel = st.sidebar.multiselect(
    "UF (vazio = todas)", all_states, default=[],
)

responsibles = ana.get_responsibles(scored)
resp_sel = st.sidebar.multiselect(
    "Operador (vazio = todos)", responsibles, default=[],
)

quality_sel = st.sidebar.multiselect(
    "Qualidade", ["Excelente", "Moderada", "Limitada"],
    default=["Excelente", "Moderada", "Limitada"],
)

active_only = st.sidebar.checkbox(
    f"Apenas estações com registro recente (≥ {CATALOG_END_YEAR - RECENT_YEARS})",
    False,
    help=(
        f"O catálogo ANAF foi compilado em março/{CATALOG_END_YEAR}; usamos "
        f"os últimos {RECENT_YEARS} anos do catálogo ({CATALOG_END_YEAR - RECENT_YEARS}"
        f"–{CATALOG_END_YEAR}) como critério de recência."
    ),
)

# Aplica filtros
df = scored.dropna(subset=["Latitude", "Longitude"]).copy()
df = df[df["NYD"].fillna(0) >= min_years]
if ufs_sel:
    df = df[df["State"].isin(ufs_sel)]
if resp_sel:
    df = df[df["Responsible"].isin(resp_sel)]
if quality_sel:
    df = df[df["quality_label"].astype(str).isin(quality_sel)]
if active_only:
    ref = pd.Timestamp(year=CATALOG_END_YEAR - RECENT_YEARS, month=1, day=1)
    df = df[df["EndDate"] >= ref]

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
recent_ref = pd.Timestamp(year=CATALOG_END_YEAR - RECENT_YEARS, month=1, day=1)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Estações no recorte", f"{len(scored):,}".replace(",", "."))
k2.metric("Exibidas após filtros", f"{len(df):,}".replace(",", "."))
k3.metric(
    "Qualidade Excelente",
    f"{(scored['quality_label'] == 'Excelente').sum():,}".replace(",", "."),
)
k4.metric(
    f"Recentes ({CATALOG_END_YEAR - RECENT_YEARS}–{CATALOG_END_YEAR})",
    f"{(scored['EndDate'] >= recent_ref).sum():,}".replace(",", "."),
    help=f"Com registro ≥ {CATALOG_END_YEAR - RECENT_YEARS}, pelo catálogo ANAF (congelado em 03/{CATALOG_END_YEAR}).",
)

# --- Classificação de qualidade explicada em destaque
st.markdown("#### Classificação de qualidade das estações")
q1, q2, q3 = st.columns(3)
with q1:
    st.markdown(
        """
        <div style="border-left: 4px solid #16a34a; padding: 8px 14px; background: #f0fdf4;">
        <b style="color:#16a34a;">● Excelente</b> &nbsp; <i>score ≥ 0,75</i><br>
        <small>Longa série, poucas falhas e dados recentes. Ideal para
        calibração e validação rigorosa.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
with q2:
    st.markdown(
        """
        <div style="border-left: 4px solid #f59e0b; padding: 8px 14px; background: #fffbeb;">
        <b style="color:#b45309;">● Moderada</b> &nbsp; <i>0,50 – 0,75</i><br>
        <small>Dados razoáveis com limitações em algum critério. Uso com
        cuidado — verificar período e falhas.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
with q3:
    st.markdown(
        """
        <div style="border-left: 4px solid #dc2626; padding: 8px 14px; background: #fef2f2;">
        <b style="color:#dc2626;">● Limitada</b> &nbsp; <i>score < 0,50</i><br>
        <small>Séries curtas, muitas falhas ou dados antigos. Usar com
        cautela; pode não ser adequada para calibração.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Score composto: **50%** anos de dados (satura em 40 anos) · **30%** "
    "completude (1 − falhas%, satura em 50%) · **20%** recência (anos "
    "desde o último registro, satura em 20). Detalhes no rodapé."
)

# ---------------------------------------------------------------------------
# Mapa pydeck Brasil
# ---------------------------------------------------------------------------
st.subheader("Mapa interativo — Brasil")

def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

_Q_COLORS = {"Excelente": "#16a34a", "Moderada": "#f59e0b", "Limitada": "#dc2626"}
df["color"] = df["quality_label"].astype(str).map(
    lambda q: _hex_to_rgb(_Q_COLORS.get(q, "#dc2626"))
)

# Destaque Manuel Duarte (58585000)
df["radius"] = 4000
is_md = df["Code"] == "58585000"
df.loc[is_md, "radius"] = 30000

# Tooltip precisa de strings
df["_start"] = df["StartDate"].dt.strftime("%Y-%m-%d").fillna("?")
df["_end"] = df["EndDate"].dt.strftime("%Y-%m-%d").fillna("?")

layers = [pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["Longitude", "Latitude"],
    get_color="color",
    get_radius="radius",
    radius_min_pixels=3,
    radius_max_pixels=10,
    pickable=True,
    opacity=0.85,
    stroked=True,
    filled=True,
    line_width_min_pixels=0.5,
)]

view = pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5, pitch=0)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={
        "html": (
            "<b>{Code} — {Name}</b><br/>"
            "{City}, {State}<br/>"
            "Operador: {Responsible}<br/>"
            "Área de drenagem: {DrainageArea} km²<br/>"
            "Período: {_start} a {_end}<br/>"
            "Anos: {NYD} | Falhas: {MD}%<br/>"
            "Qualidade: <b>{quality_label}</b>"
        ),
        "style": {"fontSize": "12px"},
    },
    map_style="light",
)

col_map, col_leg = st.columns([5, 1])
with col_map:
    st.pydeck_chart(deck, use_container_width=True, height=600)
with col_leg:
    st.markdown(
        """
        <div style="padding-top:50px; font-size:13px; line-height:2;">
        <b>Qualidade</b><br>
        <span style="color:#16a34a;">●</span> Excelente<br>
        <span style="color:#f59e0b;">●</span> Moderada<br>
        <span style="color:#dc2626;">●</span> Limitada<br>
        <br>
        <b>★ Destacada</b><br>
        58585000 — Manuel Duarte<br>(bacia do estudo)
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Estatisticas gerais
# ---------------------------------------------------------------------------
st.subheader("Estatísticas gerais")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_ana_quality_pie(scored), use_container_width=True)
with c2:
    st.plotly_chart(plot_ana_by_state(scored), use_container_width=True)

st.plotly_chart(plot_ana_nyd_distribution(scored), use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela detalhada
# ---------------------------------------------------------------------------
st.subheader("Consulta tabular")
st.caption(
    f"{len(df):,} estações após filtros. Ordene por qualquer coluna; a "
    "busca por texto funciona sobre qualquer campo.".replace(",", ".")
)

display_cols = [
    "Code", "Name", "City", "State", "Responsible",
    "DrainageArea", "StartDate", "EndDate", "NYD", "MD",
    "quality_label", "quality_score",
]
st.dataframe(
    df[display_cols].rename(columns={
        "Code": "Código", "Name": "Nome", "City": "Cidade",
        "State": "UF", "Responsible": "Operador",
        "DrainageArea": "Área drenagem (km²)",
        "StartDate": "Início", "EndDate": "Fim",
        "NYD": "Anos", "MD": "Falhas (%)",
        "quality_label": "Qualidade", "quality_score": "Score",
    }).style.format({
        "Área drenagem (km²)": "{:,.0f}",
        "Anos": "{:.0f}",
        "Falhas (%)": "{:.1f}",
        "Score": "{:.2f}",
        "Início": "{:%Y-%m-%d}",
        "Fim": "{:%Y-%m-%d}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True, height=400,
)

# Export
csv = df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar seleção (CSV)", csv,
    file_name="estacoes_convencionais_filtradas.csv",
    mime="text/csv",
)

st.divider()
st.markdown(
    """
    **Fontes dos dados:**
    - Catalogo ANAF: Carvalho, W. A., & Braga, A. S. (2020). *HydroBR: A Python
      package for hydrometeorological data acquisition from Brazilian databases*.
      Zenodo. [doi:10.5281/zenodo.3755065](https://doi.org/10.5281/zenodo.3755065)
    - Rede ANA: Agencia Nacional de Aguas e Saneamento Basico, Rede
      Hidrometeorologica Nacional (RHN), [www.gov.br/ana](https://www.gov.br/ana).

    **Nota metodologica:** a classificacao por qualidade combina tres
    componentes — 50% anos de dados (saturando em 40), 30% completude
    (1 - falhas%, saturando em 50%) e 20% recencia (anos desde o ultimo
    registro, saturando em 20). As classes sao *Excelente* (≥ 0,75),
    *Moderada* (0,50 – 0,75) e *Limitada* (< 0,50).
    """
)
