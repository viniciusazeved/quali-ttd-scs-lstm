"""
Pagina da bacia do Rio Preto (ate Manuel Duarte) — mapa das 245 ottobacias.
"""
from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import data_loader as dl
from plots import (
    plot_ottobacia_attr_histogram,
    plot_ottobacia_scatter,
)

st.title("Bacia do Rio Preto — até Manuel Duarte (58585000)")
st.caption(
    "Sub-bacia do Rio Preto — afluente da margem direita do Paraíba do Sul, "
    "discretizada em 245 ottobacias com atributos físicos derivados de "
    "MapBiomas, ANADEM e BHAE_CN-2022"
)

with st.expander("Contexto regional — hierarquia e importância"):
    st.markdown(
        """
        **Hierarquia hidrográfica:**
        Bacia do Rio Paraíba do Sul → Bacia do Rio Preto → **Sub-bacia do
        Rio Preto (até Manuel Duarte)**.

        A Bacia do Paraíba do Sul é uma das mais importantes do Brasil:
        abastece aproximadamente **14,2 milhões de pessoas** (CEIVAP, 2024)
        e sustenta expressivo parque industrial e hidrelétrico. O Rio Preto
        é afluente da margem direita, drenando áreas da **Serra da
        Mantiqueira**. A gestão integrada é coordenada pelo **CEIVAP**
        (Comitê de Integração da Bacia Hidrográfica do Rio Paraíba do Sul).

        **Características da bacia de estudo (até Manuel Duarte):**

        | Atributo | Valor |
        |---|---|
        | Código ANA | 58585000 |
        | Área de drenagem | 3.117 km² |
        | Localização | Divisa RJ/MG |
        | Elevação | 340 – 2.100 m (gradiente expressivo) |
        | Clima (Köppen) | Cwa/Cwb (subtropical de altitude) |
        | Precipitação média anual | ~1.500 mm |
        | Regime | Natural, sem barramentos significativos |
        | Nº de ottobacias | 245 |
        | Área média por ottobacia | 12,7 km² |
        | CN médio (ponderado) | 63,0 |
        | $T_c$ médio (Maidment + Manning) | 23,4 h |
        | $T_c$ máx (pixel, raster 30 m) | 105,4 h |
        | $T_c$ máx (ottobacia, Manning) | 76,6 h |
        | LULC dominante | Agropecuária 60,5% · Floresta 38,5% |
        | Vazão média (2021–2025) | 66,3 m³/s |
        | Vazão específica | 21,3 L/s/km² |

        **Por que esta bacia?** Regime natural (permite avaliar a
        resposta direta às forçantes meteorológicas, sem interferência
        de operação de reservatórios), sazonalidade marcante
        (chuvoso out–mar com Q > 100 m³/s; seco jun–set com Q < 40 m³/s)
        e série horária de 5 anos (2021–2025) com disponibilidade média
        de 91,6%.
        """
    )

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
try:
    bacia_gdf = dl.load_bacia_geom()
    ottobacias_gdf = dl.load_ottobacias_geom()
    attrs = dl.load_ottobacia_attrs()
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao carregar geometrias: {exc}")
    st.stop()

# Anexa atributos do HDF5 ao GeoDataFrame (ordem preservada entre fontes)
if len(ottobacias_gdf) == len(attrs):
    # Colunas que ainda nao estao no gdf original (evita sobrescrever cn_2022/area_km2)
    for col in attrs.columns:
        if col not in ottobacias_gdf.columns:
            ottobacias_gdf[col] = attrs[col].values

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ottobacias", f"{len(ottobacias_gdf)}")
k2.metric("Área total (km²)", f"{attrs['area_km2'].sum():,.0f}".replace(",", "."))
k3.metric("CN médio", f"{attrs['cn_2022'].mean():.1f}")
k4.metric("Tc médio (h)", f"{attrs['tc_base_h'].mean():.2f}")

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
col_top1, col_top2 = st.columns([0.55, 0.45])

with col_top2:
    st.markdown("### Controles do mapa")
    attr_choice = st.selectbox(
        "Colorir por",
        options=["cn_2022", "tc_base_h", "tc_manning_h", "area_km2"],
        format_func=lambda x: {
            "cn_2022": "Curve Number (2022)",
            "tc_base_h": "Tc — Maidment base (h)",
            "tc_manning_h": "Tc — Manning (h)",
            "area_km2": "Área (km²)",
        }[x],
    )
    cmap = st.selectbox(
        "Paleta",
        ["YlOrRd", "YlGnBu", "Viridis", "RdYlGn_r"],
        index=0,
    )
    show_labels = st.checkbox("Exibir ID das ottobacias ao passar o mouse", True)

with col_top1:
    st.markdown("### Mapa")

    # Calcula bbox para zoom
    minx, miny, maxx, maxy = ottobacias_gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Hidrografia do Paraíba do Sul (trechos que tocam ou estão próximos da bacia)
    try:
        hydro_ps = dl.load_hydro_ps()
        # Filtrar por bbox ampliada da bacia para mapa detalhado
        margem = 0.2
        bbox = (minx - margem, miny - margem, maxx + margem, maxy + margem)
        hydro_view = hydro_ps.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        folium.GeoJson(
            hydro_view.__geo_interface__,
            name="Rios (ANA)",
            style_function=lambda feat: {
                "color": "#3b82f6", "weight": 1.5, "opacity": 0.8,
            },
        ).add_to(m)
    except Exception:
        pass

    # Contorno da bacia
    folium.GeoJson(
        bacia_gdf.__geo_interface__,
        name="Contorno da bacia",
        style_function=lambda feat: {
            "color": "#1f2937", "weight": 3, "fillOpacity": 0.0,
        },
    ).add_to(m)

    # Choropleth via folium.Choropleth (mais leve que GeoJson individual)
    attr_values = ottobacias_gdf[attr_choice].tolist()
    vmin, vmax = float(min(attr_values)), float(max(attr_values))

    # id para o choropleth
    ottobacias_gdf = ottobacias_gdf.reset_index(drop=True)
    ottobacias_gdf["otto_id"] = ottobacias_gdf.index.astype(str)
    data_choro = pd.DataFrame({
        "otto_id": ottobacias_gdf["otto_id"],
        "valor": ottobacias_gdf[attr_choice],
    })

    folium.Choropleth(
        geo_data=ottobacias_gdf[["otto_id", "geometry"]].__geo_interface__,
        data=data_choro,
        columns=["otto_id", "valor"],
        key_on="feature.properties.otto_id",
        fill_color=cmap,
        fill_opacity=0.75,
        line_opacity=0.4,
        line_weight=0.5,
        legend_name=attr_choice,
        highlight=True,
    ).add_to(m)

    # Tooltip
    if show_labels:
        tooltip = folium.GeoJson(
            ottobacias_gdf[["otto_id", "area_km2", "cn_2022", "tc_base_h",
                             "tc_manning_h", "geometry"]].__geo_interface__,
            name="Tooltip",
            style_function=lambda feat: {"fillOpacity": 0.0, "weight": 0},
            highlight_function=lambda feat: {"weight": 2, "color": "#111827"},
            tooltip=folium.GeoJsonTooltip(
                fields=["otto_id", "area_km2", "cn_2022", "tc_base_h", "tc_manning_h"],
                aliases=["Ottobacia:", "Área (km²):", "CN 2022:",
                         "Tc base (h):", "Tc Manning (h):"],
                localize=True,
                sticky=False,
                labels=True,
            ),
        )
        tooltip.add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m, use_container_width=True, height=550, returned_objects=[])

# ---------------------------------------------------------------------------
# Distribuicao dos atributos
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Distribuição dos atributos físicos")

tab_cn, tab_tc, tab_area, tab_rel = st.tabs([
    "Curve Number", "Tempo de concentração", "Área", "Relações entre atributos"
])

with tab_cn:
    st.plotly_chart(
        plot_ottobacia_attr_histogram(attrs, "cn_2022", "Distribuição do CN (2022)"),
        use_container_width=True,
    )
    st.caption(
        "O Curve Number foi obtido do produto oficial **BHAE_CN-2022** da ANA "
        "(Nota Técnica nº 9/2025/COMUC/SHE), na escala 1:250.000, combinando "
        "MapBiomas Coleção 8.0 (ano-base 2022) com classificação de solos do "
        "IBGE (2023). A classificação hidrológica dos solos segue Sartori (2005), "
        "adaptada para Latossolos e Argissolos brasileiros — a classificação "
        "original do USDA tende a superestimar o escoamento em solos tropicais."
    )
    st.info(
        "CN médio = 63,0 → retenção potencial S = 149 mm. Compatível com a "
        "predominância de pastagens (60,5%) sobre solos bem desenvolvidos. "
        "No modelo, **CN é fixo** (vem do produto oficial) e **λ é aprendível** "
        "— viabiliza a transferência para bacias não monitoradas na Fase 2."
    )

with tab_tc:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_ottobacia_attr_histogram(attrs, "tc_base_h", "Tc (Maidment base)"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            plot_ottobacia_attr_histogram(attrs, "tc_manning_h", "Tc (Manning)"),
            use_container_width=True,
        )
    st.caption(
        "O $T_c$ foi calculado de forma distribuída (raster 30 m) via "
        "**WhiteboxTools** a partir do **DEM ANADEM** (IPH/UFRGS + ANA, redução de "
        "85% no viés de vegetação vs Copernicus GLO-30). Pipeline: DEM → "
        "declividade → direção de fluxo D8 → acumulação → campo de velocidades "
        "(Maidment 1996, $V = V_m \\cdot S^{0,5} \\cdot A^{0,5} / \\overline{S^{0,5} A^{0,5}}$) → "
        "integração ao longo do caminho de fluxo → estatística zonal por ottobacia."
    )
    st.info(
        "**Maidment (Base)** usa apenas declividade e área acumulada — reflete "
        "a topografia pura. **Manning** ajusta pelo coeficiente de rugosidade "
        "derivado do LULC ($n_{floresta} = 0,15$, $n_{agro} = 0,035$, "
        "$n_{urbano} = 0,015$). O Manning eleva o $T_c$ médio de 9,5 h → 23,4 h "
        "(+145%) devido às cabeceiras florestadas na Mantiqueira."
    )

with tab_area:
    st.plotly_chart(
        plot_ottobacia_attr_histogram(attrs, "area_km2", "Área das ottobacias"),
        use_container_width=True,
    )
    st.caption(
        f"Total: {attrs['area_km2'].sum():,.1f} km² — média: {attrs['area_km2'].mean():.2f} km² — "
        f"mediana: {attrs['area_km2'].median():.2f} km². A ottocodificação de Pfafstetter "
        "gera subcaptações naturalmente encaixadas na rede hidrográfica.".replace(",", ".")
    )

with tab_rel:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            plot_ottobacia_scatter(attrs, "area_km2", "cn_2022"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            plot_ottobacia_scatter(attrs, "tc_base_h", "tc_manning_h"),
            use_container_width=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Tabela
# ---------------------------------------------------------------------------
st.subheader("Tabela completa")
st.dataframe(
    attrs.round(3),
    use_container_width=True, height=300,
)
