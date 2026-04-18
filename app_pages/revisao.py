"""
Revisao bibliografica — sintese do Capitulo 2.

Seis eixos: PUB, modelos conceituais, LSTM em hidrologia, arquiteturas
temporais, modelos diferenciaveis, e restricoes fisicas.
"""
from __future__ import annotations

import streamlit as st

st.title("Revisão bibliográfica")
st.caption(
    "Capítulo 2 da qualificação · Panorama do estado da arte em modelagem "
    "chuva-vazão híbrida (2018–2026)"
)

st.markdown(
    """
    Esta seção mostra os **fundamentos teóricos** em que nos apoiamos,
    organizados em seis temas, e termina posicionando o trabalho em
    relação ao que já foi feito na literatura recente.

    **Por que isso importa:** quando a gente afirma que o modelo
    proposto "preenche 3 lacunas", precisamos mostrar exatamente
    **o que já existia antes**. A revisão não é decoração acadêmica —
    é o que justifica a novidade do trabalho.
    """
)

tab_conc, tab_lstm, tab_arch, tab_diff, tab_hib, tab_sint = st.tabs([
    "Modelos conceituais",
    "LSTM em hidrologia",
    "Arquiteturas temporais",
    "Modelos diferenciáveis",
    "SCS-CN em híbridos",
    "Síntese & posicionamento",
])

# ============================================================ Conceituais
with tab_conc:
    st.subheader("Modelos conceituais — referência operacional")

    st.markdown(
        """
        Modelos conceituais representam a bacia como sistema de
        **reservatórios interconectados**, em que cada reservatório simula
        um processo hidrológico específico (interceptação, armazenamento
        no solo, escoamento superficial e subterrâneo).

        **Referências da literatura internacional:**

        | Modelo | Origem | Características |
        |---|---|---|
        | **GR4J** | França (Perrin et al., 2003) | Apenas 4 parâmetros — parcimônia, bom desempenho em diversos climas |
        | **HBV** | Suécia (Lindström et al., 1997) | Representação explícita de neve; amplamente usado em regiões temperadas/árticas |
        | **Sacramento (SAC-SMA)** | EUA (Burnash, 1973) | 16 parâmetros; operacional no National Weather Service |
        | **MGB** | Brasil (Collischonn et al., 2007) | Semi-distribuído, conceito VIC; adaptado a bacias brasileiras |

        **Limitações reconhecidas:**

        1. **Equifinalidade** (Beven, 2001) — *muitas combinações
           diferentes de parâmetros* produzem ajustes igualmente bons
           aos dados. Ou seja: não dá para saber se os parâmetros
           calibrados refletem a realidade ou se são apenas um dos
           tantos "truques" que fazem as contas baterem.
        2. **Requerem vazão observada** para calibrar — inviável em
           bacias sem monitoramento.
        3. **Estrutura rígida** — o modelo impõe uma visão específica
           do processo hidrológico, com reservatórios e equações
           pré-definidas.
        4. **Concentrados** — a maioria trata a bacia como "caixa
           única" (*lumped*), perdendo a heterogeneidade interna.

        **Método SCS-CN** (Soil Conservation Service, 1972) permanece
        uma das ferramentas mais utilizadas para separação de escoamento,
        **sem exigir calibração com dados de vazão**. No entanto, o
        valor clássico **λ = 0,2** foi questionado por Woodward et al.
        (2003) e por ValleJunior et al. (2019), que reportaram mediana
        de **λ = 0,045** em bacias tropicais brasileiras — motivação
        direta para tratá-lo como **parâmetro aprendível** neste
        trabalho.
        """
    )

# ================================================================= LSTM
with tab_lstm:
    st.subheader("LSTM em hidrologia — a revolução de 2018")

    st.markdown(
        """
        As redes **LSTM** (*Long Short-Term Memory*, Hochreiter &
        Schmidhuber, 1997) são um tipo de **rede neural recorrente**
        — processam séries temporais passo a passo, como uma sequência
        de tempo. O que as torna especiais são os chamados **portões
        de memória** (entrada, esquecimento, saída), que permitem
        "lembrar" de informação relevante **de muito tempo atrás**. A
        partir de **2018**, transformaram a hidrologia computacional.
        """
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **Trabalhos seminais:**

            - **Kratzert et al. (2018)** — LSTMs treinadas em múltiplas
              bacias (CAMELS-US) **superam modelos conceituais** como
              SAC-SMA. NSE mediano de 0,63 em validação temporal.
            - **Kratzert et al. (2019)** — **EA-LSTM** (*Entity-Aware
              LSTM*) incorpora atributos estáticos da bacia. NSE mediano
              **0,76–0,88** → novo estado da arte para modelagem
              chuva-vazão *data-driven*.
            - **Frame et al. (2022)** — LSTMs multi-bacia preveem
              **eventos extremos** com acurácia superior a modelos
              conceituais calibrados individualmente.
            - **Lees et al. (2022)** — os **estados internos da LSTM**
              formam representações que correspondem a variáveis
              hidrológicas (umidade do solo, neve) — sem supervisão
              explícita.
            """
        )

    with col_b:
        st.markdown(
            """
            **Mas há limitações:**

            - **Caixa-preta** — os estados internos não têm correspondência
              direta com variáveis observáveis (Reichstein et al., 2019)
            - **Extrapolação para condições fora da distribuição** de
              treinamento pode degradar (Frame et al., 2022)
            - **Decisões críticas** (alerta de cheias) exigem
              interpretabilidade que a LSTM pura não oferece
              (Nearing et al., 2021)
            - **Sem garantias de conservação de massa** (Hoedt et al.,
              2021)

            **Nearing et al. (2021)** propuseram uma mudança de papel:
            ao invés de "construir modelos", os hidrólogos devem
            "fornecer **conhecimento prévio**" que guie o aprendizado.
            É o ponto de partida dos modelos **diferenciáveis**.
            """
        )

    st.info(
        "**Kratzert et al. (2024):** *LSTMs nunca devem ser treinadas em "
        "uma única bacia*. A ideia é que, quando a rede aprende com **várias "
        "bacias diferentes ao mesmo tempo**, ela é forçada a aprender "
        "**padrões gerais** (o que é comum a todas) em vez de decorar "
        "características específicas de uma bacia. É exatamente o que "
        "motiva a **Fase 2** do doutorado — treinar em ~100 bacias "
        "brasileiras."
    )

# =========================================================== Arquiteturas
with tab_arch:
    st.subheader("Arquiteturas temporais — LSTM, Transformer, TCN, Mamba")

    st.markdown(
        """
        Três famílias de arquiteturas dominam a literatura recente. A
        convergência observada em *benchmarks* recentes (Liu et al., 2025)
        reforça que **a escolha do *encoder* temporal é menos crítica
        que a qualidade da representação física e dos dados**.
        """
    )

    comp_df_md = """
    | Característica | LSTM | Transformer | TCN | Mamba |
    |---|---|---|---|---|
    | Processamento | Sequencial | Paralelo | Paralelo | Paralelo |
    | Complexidade | $O(n)$ | $O(n^2)$ | $O(n \\log n)$ | $O(n)$ |
    | Memória | Célula explícita | Atenção | Campo receptivo fixo | Estado estruturado |
    | Volume de dados | Moderado | Grande | Moderado | Moderado |
    | Integração com física | **Estabelecida** | Emergente | Limitada | Exploratória |
    | Aplicações em hidrologia | **Extensivas** | Crescentes | Limitadas | Incipientes |
    """
    st.markdown(comp_df_md)

    st.markdown(
        """
        **Justificativa da escolha (LSTM) neste trabalho:**

        1. **Amplo histórico de validação** em hidrologia desde Kratzert
           (2018/2019) — consolidada como referência
        2. **Compatibilidade estabelecida** com camadas físicas
           diferenciáveis (δHBV, DeepGR4J)
        3. **Estados internos interpretáveis** — facilita análise do
           que o modelo aprende (Lees et al., 2022)

        *Experimentos exploratórios conduzidos durante a Fase 0 (Tensor
        Hydro) com Mamba como encoder não apresentaram ganho
        significativo — consistente com a convergência observada entre
        arquiteturas temporais (Liu et al., 2025).*
        """
    )

# ============================================================= Diferenciáveis
with tab_diff:
    st.subheader("Modelos hidrológicos diferenciáveis (2022→)")

    st.markdown(
        """
        A classificação de **Shen et al. (2023)** e **Xu et al. (2025)**
        identifica quatro estratégias para combinar física e IA:

        #### 1. A IA prevê os parâmetros físicos
        Uma rede neural **estima os parâmetros** de um modelo físico a
        partir de atributos da bacia. **Tsai et al. (2021)** mostraram
        que parâmetros do SAC-SMA podem ser preditos assim, eliminando
        a calibração individual de cada bacia.

        **👉 É exatamente a ideia da Fase 2 do nosso doutorado.**

        #### 2. Trocar pedaços do modelo físico por redes neurais
        Alguns módulos (por ex., propagação de escoamento) são
        **substituídos** por redes. **He et al. (2025)** avaliaram isso
        no HBV — trocar o módulo de propagação deu mais ganho que
        trocar o módulo de geração de escoamento.

        #### 3. IA corrige os erros da física
        A rede neural atua como **pós-processador** que corrige os erros
        sistemáticos do modelo físico (Frame et al., 2022). É a
        estratégia **mais próxima do papel da LSTM no nosso modelo**.

        #### 4. Tudo treinado junto (diferenciável)
        O modelo inteiro — física + neural — é expresso como um grande
        **grafo de operações matemáticas**. Os parâmetros físicos são
        ajustados por **gradiente** (propagação reversa de erros),
        junto com os pesos da rede (Feng et al., 2022).

        **Exemplos-chave:**

        - **δHBV** (Feng et al., 2022/2024; Song et al., 2025) — HBV
          diferenciável. Ji et al. (2025, *Nature Communications*)
          demonstram padrões hidrológicos globais com parâmetros
          interpretáveis em escala continental.
        - **DeepGR4J** (Kapoor et al., 2023) — GR4J com CNN/LSTM em
          223 bacias australianas.
        - **DRRAiNN** (Scholz et al., 2025) — modelo distribuído
          totalmente diferenciável com rastreamento de fontes.
        - **TS-DUH** (Hu et al., 2024) — campo de velocidades de
          Maidment + SCS-CN, mas **com parâmetros fixos** (não
          diferenciáveis) e sem componente neural.

        **Conclusão da revisão:** a formulação de **Maidment (1996)**
        — que calcula o tempo de viagem da água a partir da topografia —
        **nunca foi implementada como camada neural com parâmetros
        aprendíveis**. Os trabalhos mais próximos usam distribuições
        gamma genéricas, sem ligação direta com a geomorfologia da
        bacia. **Essa é a primeira lacuna que nosso trabalho preenche.**
        """
    )

# =========================================================== SCS-CN híbridos
with tab_hib:
    st.subheader("SCS-CN em modelos híbridos — a segunda lacuna")

    st.markdown(
        """
        **Isik et al. (2013)** desenvolveram um dos primeiros modelos em
        que o SCS-CN alimenta uma ANN. **Merizalde et al. (2023)** usam
        o SCS-CN como pré-processador para LSTMs. **Meer et al. (2025)**
        revisam abordagens similares.

        **Contudo**, na revisão conduzida **não foi identificado** trabalho
        em que o SCS-CN seja implementado como **componente totalmente
        diferenciável** com coeficiente de abstração inicial $\\lambda$
        aprendível via *backpropagation* e CN fixo como *prior* físico.

        Os trabalhos existentes usam uma abordagem **modular**: as saídas
        do SCS-CN alimentam a rede neural como variáveis de entrada,
        **mas os gradientes não fluem através das equações do método**.
        Esta é a **segunda lacuna**.

        #### Por que λ aprendível?

        - **Banasik et al. (2014):** λ predominantemente **< 0,05** em
          bacia urbanizada polonesa
        - **Afrasiabikia et al. (2025):** λ varia entre **0,01 e 0,30**;
          calibração conjunta CN + λ melhora NSE significativamente (0,78)
        - **ValleJunior et al. (2019) + Oliveira et al. (2016):** em
          bacias tropicais brasileiras, λ **mediano = 0,045**, com
          **96,7% dos valores < 0,2**

        Fixar λ = 0,2 introduz **erro sistemático** em aplicações
        brasileiras — fazê-lo aprendível é a resposta natural.

        **No modelo proposto**, mantemos **CN fixo** (vem do produto
        oficial BHAE_CN-2022 da ANA) e tornamos **λ aprendível**. Essa
        decisão:

        - Preserva consistência com o produto oficial brasileiro
        - Viabiliza a **transferência direta** para bacias não
          monitoradas na Fase 2
        - Permite adaptação local via gradiente
        """
    )

# ============================================================ Síntese
with tab_sint:
    st.subheader("Síntese — as três lacunas preenchidas")

    st.markdown(
        """
        A revisão identifica três lacunas específicas que este trabalho
        preenche:
        """
    )

    st.markdown(
        """
        #### Lacuna 1 — TTD diferenciável baseado em Maidment

        Embora trabalhos recentes ($\\delta$HBV, DeepGR4J) utilizem
        **distribuições gamma paramétricas** como TTD diferenciável, a
        formulação baseada no **campo de velocidades de Maidment (1996)**
        — com parâmetros com **correspondência direta a grandezas
        geomorfológicas mensuráveis** ($t_{c\\_scale}$, $\\sigma$) — não
        foi identificada como camada diferenciável na literatura
        revisada.

        #### Lacuna 2 — SCS-CN com $\\lambda$ aprendível

        Embora o coeficiente de abstração inicial $\\lambda$ seja
        reconhecidamente dependente de condições locais (Brandão et al.,
        2025), os estudos revisados **não reportam implementação deste
        parâmetro como variável aprendível em camada neural
        diferenciável**.

        #### Lacuna 3 — Integração tripla TTD + SCS-CN + LSTM

        **Não foi identificada arquitetura diferenciável *end-to-end***
        que integre simultaneamente (1) hidrograma unitário distribuído
        baseado em tempo de viagem, (2) separação de escoamento via
        SCS-CN e (3) rede LSTM para refinamento residual.
        """
    )

    st.divider()

    st.markdown("### Posicionamento em relação ao estado da arte")

    st.markdown(
        """
        | Aspecto | Abordagens existentes | Modelo Proposto |
        |---|---|---|
        | **Hidrograma unitário** | Modelos concentrados (Nash, Clark) ou Maidment com parâmetros fixos | IUH gaussiano com $t_{c\\_scale}$, $\\sigma$ **aprendíveis** |
        | **Separação de escoamento** | SCS-CN tabelado ou calibrado por otim. global | SCS-CN com CN **fixo** (ANA) e $\\lambda$ **aprendível** |
        | **Papel da NN** | NN modela hidrograma completo ou substitui módulo conceitual | LSTM como **refinamento residual** da estimativa física |
        | **Formulação** | $Q = \\mathrm{NN}(\\mathrm{inputs})$ ou modelo conceitual puro | $Q = \\mathrm{LSTM}(Q_{\\text{physics}}, P, t)$ |
        | **Diferenciabilidade** | Total em recentes ($\\delta$HBV, DeepGR4J) com IUH gamma | **Totalmente diferenciável** com IUH gaussiano vinculado a Maidment |
        | **Aplicabilidade em PUB** | Requer dados locais ou falha em extrapolação | Construído para PUB desde o início (não autorregressivo + parâmetros físicos regionalizáveis) |
        """
    )

    st.success(
        "**Contribuição original:** a combinação específica — **TTD de "
        "Maidment + SCS-CN + LSTM em framework end-to-end**, com o papel "
        "explícito da LSTM como corretor residual — é, segundo esta "
        "revisão, **combinação sem precedente direto na literatura "
        "revisada (2018–2026)**."
    )

st.divider()

with st.expander("Referências principais por tema"):
    st.markdown(
        """
        | Tema | Referências principais |
        |---|---|
        | Hidrologia clássica | Tucci (2005) · Beven (2012) · Beven (2001 — equifinalidade) |
        | Hidrograma unitário | Sherman (1932) · Clark (1945) · Nash (1957) · **Maidment et al. (1996)** |
        | SCS-CN | SCS (1972) · Woodward et al. (2003) · ValleJunior et al. (2019) · Afrasiabikia et al. (2025) |
        | Arquiteturas temporais | Hochreiter & Schmidhuber (1997) · Vaswani et al. (2017) · Gu & Dao (2024 — Mamba) |
        | LSTM em hidrologia | Kratzert et al. (2018, 2019, 2024) · Frame et al. (2022) · Lees et al. (2022) · Nearing et al. (2021) |
        | Modelos diferenciáveis | Feng et al. (2022, 2024) · Shen et al. (2023) · Kapoor et al. (2023) · Xu et al. (2025) |
        | Restrições físicas | Karniadakis et al. (2021) · Frame et al. (2023) · Pokharel et al. (2023) |
        | Explicabilidade | Lundberg & Lee (2017 — SHAP) · Lees et al. (2022) · Hu et al. (2025) |
        | PUB e regionalização | Sivapalan (2003) · Blöschl et al. (2013) · Hrachowitz et al. (2013) |
        """
    )
