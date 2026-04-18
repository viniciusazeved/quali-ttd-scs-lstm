"""
Capa & Sintese — abertura da apresentacao.

Pagina de entrada com o titulo, identificacao do trabalho, resumo em 3
linhas, resultados principais e roteiro da apresentacao.
"""
from __future__ import annotations

import streamlit as st

import data_loader as dl
from plots import plot_nse_ranking

# ---------------------------------------------------------------------------
# Capa
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 2rem 1rem 1rem 1rem;">
        <p style="font-size: 0.95rem; color: #64748b; margin: 0;">
            Qualificação de Doutorado · Engenharia Civil · Recursos Hídricos
        </p>
        <h1 style="font-size: 2.2rem; margin: 0.6rem 0 0.8rem 0; line-height: 1.25;">
            Modelo Hidrológico Diferenciável para Previsão de Vazão
        </h1>
        <p style="font-size: 1.25rem; color: #1e293b; font-weight: 500; margin: 0 0 0.5rem 0;">
            Integração de Hidrograma Unitário Distribuído, SCS-CN e LSTM
        </p>
        <p style="font-size: 1rem; color: #64748b; font-style: italic; margin: 0;">
            Proposta metodológica e resultados da Fase 1 (bacia única) ·
            extensão multi-bacia planejada para a Fase 2
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([0.62, 0.38])

with col_a:
    st.markdown(
        """
        #### Em três linhas

        1. **Contexto**: previsão de vazão em rios brasileiros é limitada
           pelo monitoramento insuficiente (~2.000 estações para milhares
           de bacias) — o clássico problema das **bacias sem monitoramento**
           (*PUB — Prediction in Ungauged Basins*).
        2. **Proposta**: um modelo híbrido que combina **equações
           hidrológicas clássicas** com uma **rede neural**, treinados
           em conjunto (*ponta-a-ponta*, ou *end-to-end* no jargão). São
           três módulos: **TTD** (propagação da água pelo método de
           Maidment), **SCS-CN** (separação da chuva em escoamento) e
           uma **LSTM** (rede recorrente que corrige o que os módulos
           físicos simplificaram). O modelo **não usa vazão observada
           como entrada** — apenas chuva e atributos da bacia.
        3. **Resultado da Fase 1 (bacia única)**: **NSE = 0,84** em
           previsão de 6 horas à frente e **NSE = 0,82** em simulação
           contínua de 9 meses na sub-bacia do rio Preto (Manuel Duarte,
           3.117 km²). Os parâmetros físicos aprendidos convergem para
           valores com **interpretação hidrológica clara** — um indício
           favorável para a **Fase 2**, que vai **avaliar a generalização
           espacial** para múltiplas bacias.
        """
    )

with col_b:
    st.markdown("#### Resultados principais")
    st.metric("NSE — previsão 6h", "0,84", "Muito Bom")
    st.metric("NSE — simulação contínua", "0,82", "Muito Bom")
    st.metric("Ganho do TTD vs LSTM puro", "+35%")
    st.caption(
        "*O que é NSE?* É o coeficiente de Nash–Sutcliffe — métrica "
        "padrão em hidrologia. Varia de −∞ a 1. NSE = 1 é previsão "
        "perfeita; NSE = 0 significa que o modelo não supera usar a "
        "média histórica. Segundo Moriasi et al. (2007), **NSE ≥ 0,80 = "
        "Muito Bom**. Período de teste: 2025-03-31 → 2025-12-30, sub-bacia "
        "dividida em 245 **ottobacias** (sub-bacias menores, conforme "
        "codificação Pfafstetter da ANA)."
    )

st.divider()

# ---------------------------------------------------------------------------
# Identificação do trabalho
# ---------------------------------------------------------------------------
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        """
        **Doutorando**
        Vinícius de Azevedo Silva

        **Programa**
        Engenharia Civil — Recursos Hídricos
        DRH (Depto. de Recursos Hídricos)
        FECFAU · UNICAMP

        **Laboratório**
        LAPLA — Lab. de Planejamento Ambiental
        """
    )

with i2:
    st.markdown(
        """
        **Orientador**
        Prof. Hugo de Oliveira Fagundes

        **Coorientador**
        Prof. Edevar Luvizotto Junior

        **Data da defesa**
        22 de abril de 2026
        """
    )

with i3:
    st.markdown(
        """
        **Banca examinadora**
        Hugo Fagundes *(orientador)*
        André Rodrigues *(UFMG)*
        Paulo Tarso *(UFMS)*
        Murilo Lucas *(FT/UNICAMP)*
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Ranking dos modelos
# ---------------------------------------------------------------------------
col1, col2 = st.columns([0.55, 0.45])

with col1:
    try:
        ablation = dl.load_ablation_summary()
        st.plotly_chart(
            plot_nse_ranking(ablation, horizon="NSE_6h"),
            use_container_width=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha ao carregar sumário de ablação: {exc}")

with col2:
    st.markdown(
        """
        #### O argumento em uma frase

        > *Modelo mais simples, mas com boa base hidrológica, supera
        > arquiteturas neurais elaboradas.*

        O ranking ao lado é do **estudo comparativo** com 10 versões
        do modelo (o que a literatura técnica chama de *ablação* —
        tiramos e colocamos um módulo de cada vez para medir o peso
        de cada um). Três observações-chave:

        - **Todos os modelos com TTD** (propagação física distribuída)
          superam o **LSTM puro** (rede neural sem física) e o modelo
          concentrado (*lumped* = tratar a bacia como uma "caixa única")
        - **Parâmetros aprendíveis** dominam em previsão de curto prazo (6h)
        - **Parâmetros fixos** dominam em simulação contínua →
          surge um **compromisso** (*trade-off*) — melhorar um **piora** o outro

        A caracterização desse compromisso **não estava nas hipóteses
        originais** — é o achado mais expressivo da Fase 1 e motiva
        diretamente a Fase 2 (extensão para múltiplas bacias).
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Roteiro (sumário)
# ---------------------------------------------------------------------------
st.markdown("### Roteiro da apresentação")

st.markdown(
    """
    A apresentação segue a estrutura narrativa da qualificação. Navegue
    pelas seções no menu lateral (ordem sugerida):
    """
)

sec1, sec2, sec3 = st.columns(3)

with sec1:
    st.markdown(
        """
        **Parte I — Contexto**
        - 🏠 **Capa & Síntese** *(esta página)*
        - 💡 Introdução — o problema PUB
        - 🗺️ Monitoramento ANA
        - 🧭 Trajetória da pesquisa

        **Parte II — Fundamentação**
        - 📚 Revisão bibliográfica
        """
    )

with sec2:
    st.markdown(
        """
        **Parte III — Estudo de caso**
        - 🌿 Bacia do Rio Preto
        - 📊 Dados

        **Parte IV — Modelo**
        - ⚙️ Metodologia
        """
    )

with sec3:
    st.markdown(
        """
        **Parte V — Resultados (Fase 1)**
        - 📈 Ablação & Trade-off
        - 🔍 Explorar hidrogramas
        - 🧪 Hiperparâmetros

        **Parte VI — Próximos passos**
        - 🚀 Fase 2 & cronograma
        - 🎯 Conclusões
        """
    )

st.divider()

st.caption(
    "💡 *Este painel é uma ferramenta interativa de apoio à arguição. "
    "Durante a defesa, os resultados, mapas e hidrogramas podem ser "
    "explorados ao vivo nas abas correspondentes.*"
)
