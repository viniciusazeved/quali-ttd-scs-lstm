"""
Pagina de contexto — estacoes fluviometricas TELEMETRICAS da ANA no Brasil.

Telemetrica = sensor de nivel com transmissao automatica (horaria ou
sub-horaria), dado disponivel em tempo quase-real.

Este e o subconjunto do monitoramento que fornece o dado *horario*
exigido pelo modelo TTD-SCS-LSTM — portanto, o universo de bacias
potencialmente modelaveis em regime de previsao de cheias.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import stations as ana

st.title("Monitoramento telemetrico da ANA")
st.caption(
    "Rede fluviometrica de transmissao automatica — HidroInventario/ANA, "
    "consulta em abril/2026"
)

st.markdown(
    """
    **Telemetrica** significa sensor de nivel com transmissao automatica
    (via satelite ou GPRS), com registro **horario ou sub-horario** do
    dado bruto. E o subconjunto da rede da ANA compativel com:

    - Previsao de cheias em tempo real;
    - Modelagem de eventos extremos (resposta rapida da bacia);
    - Calibracao de modelos chuva-vazao em passo sub-diario.

    Esta pagina consulta o inventario HidroWeb filtrando por
    `tpEst=1` (fluviometrica) + `telemetrica=1`.
    """
)

st.success(
    "📌 **Telemetrica = vazao horaria.** Esta e a rede efetivamente "
    "compativel com o modelo TTD-SCS-LSTM. A estacao de referencia do "
    "estudo — Manuel Duarte (58585000) — pertence a este subconjunto."
)

# ---------------------------------------------------------------------------
# Toggle: todas vs apenas RHN (rede publica da ANA)
# ---------------------------------------------------------------------------
recorte = st.radio(
    "Recorte da rede telemetrica",
    options=["Todas as redes (publica + setor eletrico + outras)",
             "Apenas RHN — rede publica da ANA"],
    index=0,
    horizontal=True,
    help=(
        "O HidroWeb agrega estacoes telemetricas de multiplas origens: RHN "
        "(rede publica da ANA), Setor Eletrico (barragens/UHEs), acudes do "
        "semi-arido, setores regulados etc. O filtro 'Apenas RHN' mostra "
        "somente a rede publica sob gestao direta da ANA."
    ),
)
only_ana = "RHN" in recorte

try:
    tele = ana.load_telemetricas(only_ana=only_ana)
except Exception as exc:  # noqa: BLE001
    st.error(f"Nao foi possivel carregar o inventario telemetrico: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Filtros (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros do mapa")

status_sel = st.sidebar.multiselect(
    "Status operacional",
    options=ana.get_status(tele),
    default=["Ativo"] if "Ativo" in ana.get_status(tele) else [],
    help="Ativo = transmitindo; Manutencao = em reparo.",
)

all_states = ana.get_states(tele)
ufs_sel = st.sidebar.multiselect(
    "UF (vazio = todas)", all_states, default=[],
)

# Origem so faz sentido se 'Todas as redes'
if not only_ana:
    origens = ana.get_origens(tele)
    origem_sel = st.sidebar.multiselect(
        "Origem (vazio = todas)",
        origens,
        default=[],
        help="RHN = rede publica ANA; Setor Eletrico = UHEs/barragens; etc.",
    )
else:
    origem_sel = []

da_min = st.sidebar.slider(
    "Area de drenagem minima (km²)", 0, 5000, 0, 100,
    help="Filtro util para focar em bacias medias/grandes.",
)

# ---------------------------------------------------------------------------
# Aplica filtros
# ---------------------------------------------------------------------------
df = tele.dropna(subset=["Latitude", "Longitude"]).copy()
if status_sel:
    df = df[df["Status"].isin(status_sel)]
if ufs_sel:
    df = df[df["State"].isin(ufs_sel)]
if origem_sel:
    df = df[df["Origem"].isin(origem_sel)]
if da_min > 0:
    df = df[df["DrainageArea"].fillna(0) >= da_min]

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("No recorte", f"{len(tele):,}".replace(",", "."))
k2.metric("Exibidas apos filtros", f"{len(df):,}".replace(",", "."))
k3.metric(
    "Ativas",
    f"{(tele['Status'] == 'Ativo').sum():,}".replace(",", "."),
)
da_mediana = tele["DrainageArea"].dropna().median()
k4.metric(
    "Area de drenagem mediana",
    f"{da_mediana:,.0f} km²".replace(",", ".") if pd.notna(da_mediana) else "—",
    help="Mediana da area de drenagem das estacoes do recorte.",
)

# ---------------------------------------------------------------------------
# Mapa pydeck Brasil
# ---------------------------------------------------------------------------
st.subheader("Mapa interativo — Brasil")


def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# Cores por Origem (quando 'todas') ou por Status (quando RHN apenas)
if only_ana:
    _COLORS_BY = {
        "Ativo": "#16a34a",
        "Manutenção": "#f59e0b",
    }
    df["_cat"] = df["Status"].fillna("(sem status)")
else:
    _COLORS_BY = {
        "RHN": "#2563eb",                     # azul — rede ANA
        "Setor Elétrico": "#dc2626",          # vermelho — UHEs
        "Setores Regulados": "#9333ea",       # roxo
        "Açudes Semiárido": "#f59e0b",        # laranja
        "CotaOnline": "#0ea5e9",              # ciano
        "Inpe/Sivam(desativadas)": "#737373", # cinza
    }
    df["_cat"] = df["Origem"].fillna("(sem origem)")

df["color"] = df["_cat"].map(
    lambda c: _hex_to_rgb(_COLORS_BY.get(c, "#737373"))
)

# Destaque Manuel Duarte (58585000)
df["radius"] = 4000
is_md = df["Code"] == "58585000"
df.loc[is_md, "radius"] = 30000

df["_da"] = df["DrainageArea"].fillna(0).astype(float)

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
            "Rio: {River}<br/>"
            "Area: {_da} km²<br/>"
            "Origem: {Origem} · Status: {Status}<br/>"
            "Operador: {Operator}"
        ),
        "style": {"fontSize": "12px"},
    },
    map_style="light",
)

col_map, col_leg = st.columns([5, 1])
with col_map:
    st.pydeck_chart(deck, use_container_width=True, height=600)
with col_leg:
    legend_html = '<div style="padding-top:40px; font-size:13px; line-height:1.9;">'
    legend_html += "<b>Categoria</b><br>"
    for cat, color in _COLORS_BY.items():
        if cat in df["_cat"].unique():
            legend_html += f'<span style="color:{color};">●</span> {cat}<br>'
    legend_html += (
        '<br><b>★ Destacada</b><br>'
        '58585000 — Manuel Duarte<br>(bacia do estudo)'
        '</div>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estatisticas gerais
# ---------------------------------------------------------------------------
st.subheader("Composicao da rede")

c1, c2 = st.columns(2)
with c1:
    # Pie por Origem (se todas) ou por Status (se RHN)
    if not only_ana:
        src = tele.dropna(subset=["Origem"])
        counts = src["Origem"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map=_COLORS_BY,
            title="Estacoes por origem administrativa",
            hole=0.5,
        )
    else:
        counts = tele["Status"].value_counts(dropna=False).fillna(0)
        fig = px.pie(
            values=counts.values, names=counts.index,
            title="RHN — por status operacional",
            hole=0.5,
        )
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Top UFs
    counts_uf = tele.groupby("State").size().sort_values(ascending=True).tail(15)
    fig = px.bar(
        x=counts_uf.values, y=counts_uf.index,
        orientation="h",
        labels={"x": "Numero de estacoes", "y": ""},
        title="Top 15 UFs — telemetricas",
    )
    fig.update_traces(marker_color="#2563eb")
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# Distribuicao de area de drenagem
if tele["DrainageArea"].notna().sum() > 0:
    st.subheader("Distribuicao da area de drenagem")
    fig = px.histogram(
        tele, x="DrainageArea",
        nbins=60,
        log_y=True,
        title="Area de drenagem (km²) — escala log em Y",
    )
    fig.update_traces(marker_color="#2563eb", opacity=0.85)
    fig.add_vline(x=3117, line_dash="dash", line_color="#dc2626",
                  annotation_text="Rio Preto / Manuel Duarte (3.117 km²)",
                  annotation_position="top right")
    fig.update_layout(template="plotly_white", height=400,
                      xaxis_title="Area de drenagem (km²)",
                      yaxis_title="Numero de estacoes (log)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela
# ---------------------------------------------------------------------------
st.subheader("Consulta tabular")
st.caption(
    f"{len(df):,} estacoes apos filtros.".replace(",", ".")
)

display_cols = [
    "Code", "Name", "City", "State", "River",
    "DrainageArea", "Origem", "Status",
    "Responsible", "Operator", "Altitude",
]
st.dataframe(
    df[display_cols].rename(columns={
        "Code": "Codigo", "Name": "Nome", "City": "Cidade", "State": "UF",
        "River": "Rio",
        "DrainageArea": "Area drenagem (km²)",
        "Responsible": "Responsavel",
        "Operator": "Operadora",
        "Altitude": "Alt. (m)",
    }).style.format({
        "Area drenagem (km²)": "{:,.0f}",
        "Alt. (m)": "{:.0f}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True, height=400,
)

csv = df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar selecao (CSV)", csv,
    file_name="estacoes_telemetricas_filtradas.csv",
    mime="text/csv",
)

st.divider()
st.markdown(
    """
    **Fonte dos dados:** Agencia Nacional de Aguas e Saneamento Basico —
    servico HidroWeb (endpoint legado `HidroInventario`, consulta em
    abril/2026), complementado com `ListaEstacoesTelemetricas` para
    status operacional.

    **Reproducibilidade:** o shapefile local foi gerado pelo script
    `Apresentacao/shp/_export_shapes.py`, que consulta os endpoints SOAP
    publicos da ANA (sem necessidade de autenticacao). Para baixar
    **series temporais** de uma estacao especifica (nao apenas o
    inventario), e preciso credencial da API HidroWebService v2.
    """
)
