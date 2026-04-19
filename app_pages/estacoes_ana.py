"""
Pagina de contexto — estacoes fluviometricas CONVENCIONAIS da ANA no Brasil.

Convencional = leitura diaria em regua/linigrafo, registro manual.
Serve como ponto de partida da narrativa: apesar da rede densa, a operacao
convencional fornece apenas vazao DIARIA, incompativel com modelagem de
eventos de cheia e previsao horaria.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import stations as ana
from plots import plot_ana_by_state, plot_ana_nyd_distribution
from embed_mode import maybe_hide_chrome

maybe_hide_chrome()

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
    scored = ana.load_convencionais(only_ana=only_ana)
except Exception as exc:  # noqa: BLE001
    st.error(f"Nao foi possivel carregar o catalogo convencional: {exc}")
    st.stop()

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

# Aplica filtros
df = scored.dropna(subset=["Latitude", "Longitude"]).copy()
df = df[df["NYD"].fillna(0) >= min_years]
if ufs_sel:
    df = df[df["State"].isin(ufs_sel)]
if resp_sel:
    df = df[df["Responsible"].isin(resp_sel)]

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
n_ana = int((scored["Responsible"] == "ANA").sum())
n_outros = int((scored["Responsible"] != "ANA").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Estações no recorte", f"{len(scored):,}".replace(",", "."))
k2.metric("Exibidas após filtros", f"{len(df):,}".replace(",", "."))
k3.metric(
    "Operadas pela ANA",
    f"{n_ana:,}".replace(",", "."),
    help="Estações com Responsible=ANA no catálogo ANAF — rede pública federal.",
)
k4.metric(
    "Demais operadoras",
    f"{n_outros:,}".replace(",", "."),
    help=(
        "Estaduais (DAEE-SP, IGAM-MG, INEA-RJ), concessionárias (COPEL, "
        "FURNAS, CEMIG) e outros agentes reguladores/operadores."
    ),
)

# ---------------------------------------------------------------------------
# Mapa pydeck Brasil
# ---------------------------------------------------------------------------
st.subheader("Mapa interativo — Brasil")

def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

# Cores: ANA (rede publica federal) vs demais operadoras
_COLOR_ANA = _hex_to_rgb("#2563eb")         # azul — ANA
_COLOR_OUTROS = _hex_to_rgb("#737373")      # cinza — demais operadoras
df["color"] = df["Responsible"].map(
    lambda r: _COLOR_ANA if r == "ANA" else _COLOR_OUTROS
)

# Separar Manuel Duarte para layer proprio acima dos demais
df["radius"] = 4000
is_md = df["Code"] == "58585000"
df_base = df[~is_md].copy()
df_md = df[is_md].copy()

# Tooltip precisa de strings
for subset in (df_base, df_md):
    subset["_start"] = subset["StartDate"].dt.strftime("%Y-%m-%d").fillna("?")
    subset["_end"] = subset["EndDate"].dt.strftime("%Y-%m-%d").fillna("?")

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=df_base,
        get_position=["Longitude", "Latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.85,
        stroked=True,
        filled=True,
        line_width_min_pixels=0.5,
    ),
    # Halo externo ao redor de Manuel Duarte (anel amarelo, sem fill)
    pdk.Layer(
        "ScatterplotLayer",
        data=df_md,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[234, 179, 8, 220],  # amarelo
        get_radius=70000,
        radius_min_pixels=18,
        radius_max_pixels=28,
        stroked=True,
        filled=False,
        line_width_min_pixels=3,
        pickable=False,
    ),
    # Ponto destacado de Manuel Duarte (por cima de tudo)
    pdk.Layer(
        "ScatterplotLayer",
        data=df_md,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[234, 179, 8, 255],  # amarelo vivo
        get_line_color=[0, 0, 0, 255],
        get_radius=42000,
        radius_min_pixels=10,
        radius_max_pixels=16,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        pickable=True,
    ),
]

view = pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5, pitch=0)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={
        "html": (
            "<b>{Code} — {Name}</b><br/>"
            "{City}, {State}<br/>"
            "Operador: <b>{Responsible}</b><br/>"
            "Área de drenagem: {DrainageArea} km²<br/>"
            "Período: {_start} a {_end}<br/>"
            "Anos de dados: {NYD} | Falhas: {MD}%"
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
        <b>Operadora</b><br>
        <span style="color:#2563eb;">●</span> ANA<br>
        <span style="color:#737373;">●</span> Outras<br>
        <br>
        <b>Destacada</b><br>
        <span style="color:#eab308; font-size: 1.3em;">◉</span>
        58585000<br>Manuel Duarte<br>
        <small>(bacia do estudo)</small>
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
    # ANA vs demais operadoras
    cat = scored["Responsible"].where(scored["Responsible"] == "ANA", "Outras")
    counts = cat.value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={"ANA": "#2563eb", "Outras": "#737373"},
        title="ANA vs demais operadoras",
        hole=0.5,
    )
    fig_pie.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
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
]
st.dataframe(
    df[display_cols].rename(columns={
        "Code": "Código", "Name": "Nome", "City": "Cidade",
        "State": "UF", "Responsible": "Operador",
        "DrainageArea": "Área drenagem (km²)",
        "StartDate": "Início", "EndDate": "Fim",
        "NYD": "Anos", "MD": "Falhas (%)",
    }).style.format({
        "Área drenagem (km²)": "{:,.0f}",
        "Anos": "{:.0f}",
        "Falhas (%)": "{:.1f}",
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

    **Nota:** o catalogo ANAF foi compilado em marco/2020 e reune as estacoes
    fluviometricas com dados registrados e disponiveis via HidroWeb naquela
    data. O campo *Operador* indica quem responde pela estacao — nem sempre
    quem a opera em campo (ex.: estacoes ANA sao frequentemente operadas
    pelo Servico Geologico do Brasil, CPRM-SGB).
    """
)
