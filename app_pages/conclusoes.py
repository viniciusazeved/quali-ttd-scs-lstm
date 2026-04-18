"""
Conclusoes da apresentacao — sintese final.
"""
from __future__ import annotations

import streamlit as st

import data_loader as dl

st.title("Conclusões")
st.caption(
    "Síntese da Fase 1 e mensagem central da qualificação"
)

# ---------------------------------------------------------------------------
# Highlights quantitativos
# ---------------------------------------------------------------------------
try:
    abl = dl.load_ablation_summary()
    cont = dl.load_continuous_summary()
    best_f = abl.iloc[0]
    best_c = cont.iloc[0]
except Exception:
    best_f = {"NSE_6h": 0.84, "Modelo": "LSTM_TTD_Base"}
    best_c = {"NSE": 0.82, "Modelo": "LSTM_TTD_Base_Fixed"}

col1, col2, col3, col4 = st.columns(4)
col1.metric("NSE — previsão 6h", f"{best_f['NSE_6h']:.2f}")
col2.metric("NSE — simulação contínua", f"{best_c['NSE']:.2f}")
col3.metric("Ganho TTD vs LSTM puro", "+35%")
col4.metric("Parâmetros aprendidos interpretáveis", "✅")

st.divider()

# ---------------------------------------------------------------------------
# Quatro conclusões-chave
# ---------------------------------------------------------------------------
st.subheader("Quatro conclusões da Fase 1")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        #### 1. TTD é o componente-chave

        A **propagação temporal distribuída** (TTD baseado em Maidment)
        é responsável pelo **maior ganho de desempenho** em relação ao
        LSTM de referência — **+35% em NSE**. Resultado consistente com
        a teoria de Maidment (1996): a representação explícita dos
        tempos de viagem é o componente mais relevante para previsão
        de vazão.

        #### 2. Trade-off otimização × generalização

        - **Parâmetros aprendíveis** → melhor previsão multi-horizonte
          (NSE 6h = 0,84)
        - **Parâmetros fixos** → melhor simulação contínua
          (NSE = 0,82)

        Este achado não estava entre as hipóteses originais e é o
        **resultado mais expressivo** da Fase 1, com implicação prática
        direta na escolha da configuração por aplicação.
        """
    )

with c2:
    st.markdown(
        """
        #### 3. Parâmetros interpretáveis

        Os parâmetros aprendidos pelos modelos diferenciáveis convergem
        para valores **fisicamente interpretáveis**:

        - $\\lambda \\approx 0{,}06$ – 0,17 (compatível com ValleJunior
          et al., 2019: mediana 0,045 em bacias tropicais)
        - $t_{c\\_scale} \\approx 1{,}2$ – 1,3 (ligeiro aumento do Tc
          calculado, indicando leve subestimação da formulação empírica)
        - $\\sigma \\approx 4$ – 5 h (dispersão coerente com a variabilidade
          dos Tc entre ottobacias)

        Validam a abordagem diferenciável: o modelo **não apenas otimiza
        métricas**, preserva correspondência com grandezas observáveis.

        #### 4. SCS-CN limitado em bacia única

        Em bacia única e com *lookback* de 240 h, a **LSTM já captura
        implicitamente** a transformação chuva-escoamento. A manutenção
        do SCS-CN na arquitetura se justifica pela **Fase 2**: CN
        atuará como ***prior* diferenciador entre bacias** com
        características pedológicas distintas.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Mensagem central
# ---------------------------------------------------------------------------
st.subheader("Mensagem central")

st.success(
    "**Base hidrológica sólida vale mais que rede neural elaborada.** "
    "O TTD-SCS-LSTM — arquitetura mais **simples** mas com equações "
    "hidrológicas clássicas embutidas — **superou** a abordagem "
    "exploratória do Tensor Hydro (PatchCNN + FiLM + Transformer) "
    "tanto em desempenho (NSE 0,84 horário vs 0,72 diário) quanto em "
    "**interpretabilidade** e **potencial de transferência para outras "
    "bacias**."
)

st.markdown(
    """
    O percurso ilustra a tese central da qualificação:

    > *Quando se troca complexidade arquitetural por base hidrológica
    > bem escolhida, o modelo fica menor, mais preciso e mais transferível.
    > A física certa pesa mais que a rede neural mais elaborada.*

    Essa não é apenas uma escolha de modelagem — é um posicionamento
    alinhado ao programa de **Nearing et al. (2021)**: o papel da
    ciência hidrológica hoje é menos "construir novos modelos" e mais
    "fornecer conhecimento prévio que guie o aprendizado de máquina".
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Contribuição em 3 planos
# ---------------------------------------------------------------------------
st.subheader("Contribuições em três planos")

col_m, col_s, col_a = st.columns(3)

with col_m:
    st.markdown(
        """
        #### Metodológico

        Uma **arquitetura hidrológica diferenciável** *end-to-end*,
        com três componentes integrados:

        1. TTD baseado em Maidment como camada diferenciável
           ($t_{c\\_scale}$, $\\sigma$ aprendíveis)
        2. SCS-CN diferenciável ($\\lambda$ aprendível em
           $[0{,}01;\\,0{,}40]$)
        3. LSTM como **refinamento residual** (não como modelo)

        **Modularidade** permite substituição de cada componente sem
        afetar o framework.
        """
    )

with col_s:
    st.markdown(
        """
        #### Científico

        Demonstração quantitativa de que:

        - A **propagação temporal distribuída** é o principal
          diferencial do modelo (+35%)
        - Parâmetros físicos **aprendíveis convergem** para valores
          interpretáveis compatíveis com a literatura brasileira
        - Existe um **trade-off sistemático** entre otimização e
          generalização em modelos diferenciáveis — caracterização
          original desta pesquisa

        Três lacunas da literatura (2018–2026) foram preenchidas.
        """
    )

with col_a:
    st.markdown(
        """
        #### Aplicado

        Dois produtos operacionais baseados em precipitação e
        atributos fisiográficos — **aplicáveis a bacias não
        monitoradas**:

        1. **Simulação contínua** para análises de disponibilidade
           hídrica, outorga e dimensionamento
        2. **Previsão de curto prazo** (multi-horizonte 1–24h) para
           sistemas de alerta e apoio à defesa civil

        Análise de cenários de mudança de uso do solo via CN e
        coeficiente de Manning explícitos.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Próximos passos
# ---------------------------------------------------------------------------
st.subheader("O que vem a seguir — Fase 2")

st.info(
    "A **Fase 2** estende o modelo para **~100 bacias brasileiras** com "
    "validação leave-one-basin-out, usando um encoder MLP para predizer "
    "os parâmetros físicos a partir de atributos da bacia. É a validação "
    "definitiva da hipótese central: **é possível generalizar para bacias "
    "não monitoradas com parâmetros físicos regionalizáveis**."
)

st.divider()

# ---------------------------------------------------------------------------
# Reconhecimentos
# ---------------------------------------------------------------------------
st.markdown(
    """
    #### Reconhecimentos

    **Orientação:** Prof. Hugo de Oliveira Fagundes (DRH/FECFAU/UNICAMP)
    **Coorientação:** Prof. Edevar Luvizotto Junior (DRH/FECFAU/UNICAMP)
    **Laboratório:** LAPLA — Laboratório de Planejamento Ambiental
    **Banca:** Hugo Fagundes · André Rodrigues (UFMG) · Paulo Tarso
    (UFMS) · Murilo Lucas (FT/UNICAMP)

    **Agradecimentos:** Programa de pós-graduação em Engenharia Civil
    (Recursos Hídricos) / UNICAMP · Agência Nacional de Águas e
    Saneamento Básico (dados de vazão, BHAE_CN-2022) · CPTEC/INPE
    (dados MERGE) · IPH/UFRGS (DEM ANADEM) · MapBiomas (uso do solo).

    ---

    **Obrigado.**
    """
)
