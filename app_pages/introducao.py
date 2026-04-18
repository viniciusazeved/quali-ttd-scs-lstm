"""
Introducao — Contexto e Motivacao (Capitulo 1 da quali).

O problema da Previsao em Bacias Nao Monitoradas (PUB).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

st.title("Introdução — o problema das bacias não monitoradas")
st.caption("Capítulo 1 da qualificação · Contexto e motivação operacional")

# ---------------------------------------------------------------------------
# Tese em 1 parágrafo
# ---------------------------------------------------------------------------
st.markdown(
    """
    > A previsão de vazão atende a **duas demandas operacionais complementares**:
    > (i) **simulação contínua** para gestão de disponibilidade hídrica,
    > dimensionamento e outorga; e (ii) **previsão de curto prazo** para
    > sistemas de alerta e defesa civil. Ambas esbarram no mesmo gargalo:
    > a maioria das bacias brasileiras simplesmente **não tem monitoramento
    > suficiente** para calibrar modelos hidrológicos tradicionais.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# O paradoxo do monitoramento
# ---------------------------------------------------------------------------
st.subheader("O paradoxo do monitoramento hidrológico brasileiro")

col1, col2 = st.columns([0.5, 0.5])

with col1:
    st.markdown(
        """
        **Importância estratégica:**
        - **Matriz elétrica** brasileira é majoritariamente hidráulica
          → requer estimativas contínuas de disponibilidade hídrica
        - **Populações vulneráveis** a cheias e secas dependem de
          sistemas de alerta com antecedência de horas a dias
        - **Gestão integrada** de recursos hídricos, outorga,
          operação de reservatórios em cascata

        **Realidade operacional:**
        - A Rede Hidrometeorológica Nacional opera **~2.000 estações
          fluviométricas** (ANA, 2024) — para um território continental
          com **milhares** de bacias catalogadas
        - Globalmente, apenas uma **pequena fração** das bacias tem
          monitoramento sistemático de vazão (Nearing et al., 2024)
        - Forte correlação entre disponibilidade de dados e PIB: as
          regiões mais **vulneráveis a inundações** são justamente as
          **menos monitoradas**
        """
    )

with col2:
    # Números com fonte verificável
    n_estacoes = 2000        # ANA (2024) — RHN
    n_ottos_n5 = 16559       # BHO/ANA nível 5 Pfafstetter (GPKG oficial usado no projeto)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Ottobacias nível 5 (BHO/ANA)", "Estações fluviométricas (RHN/ANA)"],
        x=[n_ottos_n5, n_estacoes],
        orientation="h",
        marker_color=["#cbd5e1", "#2563eb"],
        text=[f"{n_ottos_n5:,}".replace(",", "."),
              f"~{n_estacoes:,}".replace(",", ".")],
        textposition="outside",
    ))
    cov_pct = 100 * n_estacoes / n_ottos_n5
    fig.update_layout(
        title=f"Cobertura: apenas ~{cov_pct:.0f}% das ottobacias N5 têm estação",
        template="plotly_white",
        xaxis_title="Número de unidades",
        xaxis=dict(range=[0, n_ottos_n5 * 1.2]),
        height=280,
        showlegend=False,
        margin=dict(l=0, r=30, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Fontes:** ANA (2024) para a RHN (Rede Hidrometeorológica Nacional) · "
        "BHO/ANA 2017 v_01_05 para as ottobacias **nível 5** Pfafstetter "
        "(referência em escala nacional, 16.559 unidades). "
        "*Nota:* a bacia de estudo (Rio Preto / Manuel Duarte) usa uma "
        "discretização **mais fina** — 245 ottobacias de níveis 7–9, com "
        "área média de 12,7 km². São todas produtos da mesma BHO/ANA, "
        "em resoluções diferentes. "
        "O contraste aqui ilustra o problema PUB em escala nacional: "
        "calibração tradicional de modelos é inviável quando o número de "
        "bacias supera em ordens de grandeza o número de estações."
    )

st.divider()

# ---------------------------------------------------------------------------
# PUB — Década IAHS
# ---------------------------------------------------------------------------
st.subheader("PUB — Prediction in Ungauged Basins")

st.markdown(
    """
    **PUB** (Previsão em Bacias Não Monitoradas) foi tema central da
    **Década Hidrológica da IAHS (2003–2012)**, mobilizando esforços
    globais de pesquisa (Sivapalan, 2003). A síntese da década
    (Blöschl et al., 2013) identificou três estratégias tradicionais:
    """
)

colA, colB, colC = st.columns(3)
with colA:
    st.markdown(
        """
        #### Regionalização de parâmetros

        Estabelece relações estatísticas entre **parâmetros calibrados**
        de modelos hidrológicos e **atributos físicos** de bacias
        monitoradas. Técnicas modernas de *machine learning* (Random
        Forest, SVR) podem superar regressões lineares tradicionais
        (Hrachowitz et al., 2013).
        """
    )
with colB:
    st.markdown(
        """
        #### Similaridade física

        Transfere parâmetros de **bacias doadoras** selecionadas por
        proximidade em um espaço de atributos físico-climáticos. A
        eficácia depende da escolha das métricas de similaridade e
        da representatividade da rede de monitoramento.
        """
    )
with colC:
    st.markdown(
        """
        #### Proximidade espacial

        Assume que bacias **geograficamente próximas** compartilham
        características hidrológicas similares. Falha quando bacias
        vizinhas divergem em uso do solo ou geologia — comum no
        mosaico brasileiro.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Novo paradigma (2018+)
# ---------------------------------------------------------------------------
st.subheader("O novo paradigma (2018→hoje): aprendizado profundo multi-bacia")

st.markdown(
    """
    A partir de 2018, emergiu uma **quarta via**: redes neurais treinadas
    simultaneamente em **centenas de bacias**. **Kratzert et al. (2019)**
    demonstraram que **LSTMs treinadas em 531 bacias** da base CAMELS-US
    **generalizam** para bacias não vistas durante o treinamento — com
    desempenho comparável a modelos calibrados localmente.

    Em vez de *transferir parâmetros*, o modelo **aprende relações universais**
    entre forçantes meteorológicas, atributos de bacia e resposta hidrológica.
    É uma mudança qualitativa, não só quantitativa.

    **Mas há um custo.** Modelos que aprendem **só a partir dos dados**
    (sem física embutida) são criticados por:

    - **"Caixa-preta"** — não dá pra saber *por que* a rede prevê aquele
      valor: os milhões de pesos internos não têm correspondência direta
      com variáveis hidrológicas (Reichstein et al., 2019)
    - **Extrapolação limitada** para condições não vistas no treinamento
      (ex.: chuva muito acima da média histórica). Reichert et al. (2024)
    - **Ausência de garantias físicas** — nada impede que o modelo
      preveja mais vazão do que chuva, o que seria um absurdo (violação
      de conservação de massa; Hoedt et al., 2021)
    - Para **gestores e reguladores**, essa opacidade dificulta a adoção
      em decisões críticas (Nearing et al., 2021)

    Esta tensão motiva uma **terceira via** — modelos hidrológicos
    **diferenciáveis**: combinam a **interpretabilidade** dos modelos
    clássicos com a **flexibilidade de aprendizado** das redes neurais.
    *Diferenciável* aqui significa que é possível "medir" o quanto cada
    parâmetro (físico ou neural) contribuiu para o erro da previsão e
    **ajustar todos juntos**, em um único processo de treinamento.
    """
)

st.info(
    "**Premissa que atravessa todo este trabalho:** se o problema é **PUB**, "
    "por definição o modelo **não pode** depender de vazão observada em "
    "tempo real — afinal, ela não existe nessa bacia. A única entrada "
    "admissível é **P → Q** (precipitação + atributos fisiográficos). "
    "Modelos autorregressivos ($Q_{obs}$ como entrada) estão **excluídos** "
    "por premissa operacional."
)

st.divider()

# ---------------------------------------------------------------------------
# Hipóteses centrais
# ---------------------------------------------------------------------------
st.subheader("Hipótese central do trabalho")

st.markdown(
    """
    > **A integração de módulos hidrológicos diferenciáveis (TTD + SCS-CN)
    > com parâmetros aprendíveis via *backpropagation*, combinados com
    > uma LSTM de correção residual, produz um modelo interpretável,
    > generalizável e com desempenho superior a abordagens puramente
    > *data-driven* ou puramente conceituais para previsão de vazão.**

    Essa hipótese central é decomposta em **5 hipóteses operacionais**
    (H1–H5) avaliadas sistematicamente pelo estudo de ablação descrito
    no Capítulo 3 e avaliado no Capítulo 4 (ver páginas Metodologia e
    Resultados).
    """
)

st.divider()

st.markdown(
    """
    #### O que vem a seguir nesta apresentação

    - **Monitoramento ANA** — quais e onde estão as estações fluviométricas
      brasileiras (mapa interativo do catálogo ANAF)
    - **Trajetória da pesquisa** — como chegamos ao TTD-SCS-LSTM (spoiler:
      o primeiro modelo, Tensor Hydro, não generalizou)
    - **Revisão bibliográfica** — conceitos-chave, paradigmas e as 3
      lacunas que este trabalho preenche
    - **Bacia do Rio Preto** — área de estudo (sub-bacia até Manuel Duarte)
    """
)

st.caption(
    "**Referências-chave:** Sivapalan (2003) · Blöschl et al. (2013) · "
    "Hrachowitz et al. (2013) · Kratzert et al. (2018, 2019) · "
    "Reichstein et al. (2019) · Nearing et al. (2021, 2024) · "
    "Shen et al. (2023) · Reichert et al. (2024) · ANA (2024)."
)
