"""
Proximos passos — Fase 2, cronograma e publicacoes.

Sintese do Capitulo 5 (cronograma.tex) da quali, em tom didatico.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

st.title("Próximos passos — Fase 2")
st.caption(
    "O teste decisivo: o modelo funciona em uma bacia. Será que funciona "
    "em 100?"
)

# ---------------------------------------------------------------------------
# Contexto — o que a Fase 1 deixou pra Fase 2
# ---------------------------------------------------------------------------
st.markdown(
    """
    A Fase 1 mostrou que o modelo **funciona muito bem numa bacia** — NSE
    = 0,84 em previsão de 6 h e NSE = 0,82 em simulação contínua no rio
    Preto (Manuel Duarte). Mas a pergunta que fica é:

    > **Isso se repete em bacias onde o modelo nunca foi treinado?**

    Essa é a pergunta central do **problema PUB** (bacias sem
    monitoramento) — e é exatamente onde o **Tensor Hydro**
    (arquitetura anterior) falhou. Vale a pena relembrar:
    """
)

col_th1, col_th2 = st.columns(2)
with col_th1:
    st.error(
        "🛑 **Tensor Hydro (Fase 0) — o que deu errado**\n\n"
        "NSE ~0,79 na estação de treino; caiu para ~0,20–0,40 em "
        "estações que o modelo não tinha visto. O modelo aprendeu a "
        "**replicar a relação chuva-vazão daquela estação específica** "
        "— não aprendeu hidrologia."
    )
with col_th2:
    st.success(
        "✅ **TTD-SCS-LSTM (Fase 1) — o que mudou**\n\n"
        "Embutiu a física diretamente no modelo (tempo de viagem, "
        "separação de escoamento). Os parâmetros aprendidos têm "
        "**interpretação hidrológica clara** e podem, em tese, ser "
        "**inferidos a partir de atributos da bacia**."
    )

st.info(
    "**A hipótese da Fase 2 é exatamente essa:** como os parâmetros "
    "físicos do TTD e do SCS-CN têm significado hidrológico claro, "
    "eles devem ser **previsíveis a partir das características de "
    "qualquer bacia** (área, declividade, uso do solo, regime climático). "
    "Os resultados da Fase 1 — NSE = 0,82 na simulação contínua **com "
    "parâmetros físicos fixos da literatura** — indicam que esse caminho "
    "é promissor: valores razoáveis derivados de atributos já produzem "
    "resultados sólidos, sem calibração local."
)

st.divider()

# ---------------------------------------------------------------------------
# A ideia da regionalização — explicada em analogia
# ---------------------------------------------------------------------------
st.subheader("A ideia: regionalização via codificador neural")

col_a, col_b = st.columns([0.55, 0.45])

with col_a:
    st.markdown(
        r"""
        Em vez de aprender um conjunto de parâmetros para **cada bacia
        individualmente** (o que seria impossível onde não há dados de
        vazão), colocamos uma **pequena rede neural** (chamada
        *codificador* — ou *encoder*) que **aprende a prever os
        parâmetros físicos a partir dos atributos da bacia**:
        """
    )

    st.latex(r"[\,t_{c\_scale},\;\sigma,\;\lambda\,] = \mathrm{Codificador}(\mathbf{atributos}_{\text{bacia}})")

    st.markdown(
        """
        **Atributos candidatos** (disponíveis para qualquer bacia do
        Brasil, sem exigir estação fluviométrica):

        - Área de drenagem
        - Elevação média e declividade média
        - CN médio (do produto BHAE da ANA)
        - Frações de uso do solo (MapBiomas)
        - Precipitação média anual
        - Índice de aridez
        - Densidade de drenagem

        Durante o treinamento multi-bacia, o codificador **aprende o
        mapa** entre "características visíveis" e "parâmetros físicos
        internos". Se esse mapa for consistente entre bacias, o modelo
        generaliza para qualquer ponto do Brasil.
        """
    )

with col_b:
    # Diagrama conceitual
    st.markdown("**Fluxo conceitual da Fase 2:**")
    st.markdown(
        """
        ```
        Atributos da bacia X
               │
               ▼
           [Codificador]  ← aprende com 99 bacias
               │
               ▼
        Parâmetros físicos estimados
        (tc_scale, σ, λ)
               │
               ▼
          [TTD + SCS-CN]
               │
               ▼
             [LSTM]
               │
               ▼
         Q previsto na bacia X
         (sem ter usado Q de X no treino!)
        ```
        """
    )
    st.caption(
        "O codificador é a peça nova — todo o resto (TTD, SCS-CN, "
        "LSTM) é o mesmo modelo validado na Fase 1."
    )

st.divider()

# ---------------------------------------------------------------------------
# Três etapas
# ---------------------------------------------------------------------------
st.subheader("Como vamos executar a Fase 2")

col_et1, col_et2, col_et3 = st.columns(3)

with col_et1:
    st.markdown(
        """
        #### 1️⃣ Selecionar ~100 bacias

        Bacias brasileiras com:
        - Série horária de vazão via **ANA/HidroWeb**
        - Cobertura de precipitação **MERGE** (todo o Brasil desde 2000)
        - Atributos geoespaciais (BHAE, MapBiomas, DEM) — disponíveis
          para o Brasil inteiro
        - **Distribuição geográfica representativa** (diferentes biomas,
          climas, regimes)

        > Já temos uma infraestrutura automatizada para baixar dados
        > da ANA em lote. Meta realista: Nordeste semiárido, Cerrado,
        > Mata Atlântica, Sul, Amazônia.
        """
    )

with col_et2:
    st.markdown(
        """
        #### 2️⃣ Treinar em 99, testar na 100ª

        O protocolo clássico do problema PUB chama-se
        ***leave-one-basin-out***:

        - Treina o modelo em **99 bacias**
        - Testa na 100ª — que o modelo **nunca viu**
        - Repete para cada uma das 100 bacias (cada uma vira "teste"
          uma vez)
        - A métrica final é a **média em todas as bacias não vistas**

        É o teste mais severo possível — simula **exatamente** a
        situação de uma bacia nova, sem histórico.

        *(Variante: dividir em grupos geográficos — k-fold espacial.)*
        """
    )

with col_et3:
    st.markdown(
        """
        #### 3️⃣ Comparar com o estado da arte

        Não adianta chegar a NSE = 0,70 sem saber se é bom ou ruim
        no contexto internacional. Por isso vamos comparar com:

        - **EA-LSTM** (Kratzert et al., 2019) — a referência
          mundial para LSTM multi-bacia
        - **δHBV** (Feng et al., 2024) — o principal modelo
          diferenciável da literatura
        - **NeuralHydrology** — framework open-source de referência

        E também **por região climática** — entender se o modelo
        funciona melhor em certos biomas que outros.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Por que os sinais iniciais são promissores
# ---------------------------------------------------------------------------
st.subheader("Por que os sinais iniciais são promissores")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        """
        #### 🎯 Parâmetros físicos fixos já funcionam bem

        O **LSTM_TTD_Base_Fixed** — com parâmetros padrão da literatura
        ($\\lambda = 0{,}2$, $t_{c\\_scale} = 1$, $\\sigma = 3$ h) —
        alcançou **NSE = 0,82 em simulação contínua**. Isso mostra que
        valores razoáveis derivados de atributos da bacia, sem
        calibração local, já produzem bom desempenho.

        É o primeiro indício forte de transferibilidade.
        """
    )

with col_p2:
    st.markdown(
        """
        #### 🧭 Parâmetros aprendidos têm interpretação física

        Os valores convergidos ($\\lambda \\approx 0{,}06$–0,17;
        $t_{c\\_scale} \\approx 1{,}2$–1,3) são **compatíveis com a
        literatura brasileira** (ValleJunior et al., 2019) e com a
        topografia da bacia.

        Ou seja: o modelo **não está inventando valores aleatórios**
        — está convergindo para grandezas fisicamente plausíveis,
        que podem ser relacionadas a atributos mensuráveis.
        """
    )

with col_p3:
    st.markdown(
        """
        #### 🛰️ Escala horária + não autorregressivo já é avanço

        Os benchmarks internacionais de regionalização (EA-LSTM,
        δHBV) operam tipicamente em **escala diária** e/ou **usando
        vazão observada como entrada**.

        Nosso modelo opera em **resolução horária** e **apenas com
        precipitação** — condições muito mais exigentes. Os resultados
        em uma bacia já se comparam favoravelmente a esses
        benchmarks; a extensão multi-bacia é o próximo passo natural.
        """
    )

st.success(
    "**A Fase 2 é onde essas evidências se consolidam.** O protocolo "
    "*leave-one-basin-out* testa exatamente a transferibilidade para "
    "bacias nunca vistas, com comparação direta a EA-LSTM e δHBV. "
    "Se confirmada a hipótese, o modelo entrega algo inédito: "
    "previsão horária baseada só em precipitação, aplicável a "
    "qualquer bacia brasileira sem calibração local."
)

st.divider()

# ---------------------------------------------------------------------------
# Aprimoramentos planejados
# ---------------------------------------------------------------------------
st.subheader("Melhorias adicionais já planejadas")

tab_art1, tab_fase2_med = st.tabs([
    "Curto prazo — Artigo 1",
    "Médio prazo — durante a Fase 2",
])

with tab_art1:
    st.markdown(
        """
        **Melhorias a incorporar antes de submeter o Artigo 1 (M7–M8):**

        - **Trocar o formato do hidrograma unitário** — hoje usamos uma
          curva gaussiana (simétrica), mas o hidrograma real é
          **assimétrico** (sobe rápido, desce devagar). Vamos testar a
          distribuição gamma (clássica em hidrologia) e o histograma
          discreto de tempos de viagem.

        - **Regularização por conservação de massa** — adicionar na
          função de perda um termo que penaliza quando
          $\\sum Q_{pred} \\ne \\sum P_{efetiva}$ — garante que o modelo
          respeita balanço hídrico.

        - **Análise de incerteza** — treinar com múltiplas sementes
          aleatórias (*seeds*) e reportar intervalos de confiança em
          vez de um único número. Mais honesto estatisticamente.

        - **Busca refinada de hiperparâmetros com Optuna** — substituir
          a busca exaustiva em grade (108 configs) por otimização
          bayesiana via [Optuna](https://optuna.org/), que encontra
          boas configurações com menos treinamentos. *Para mais
          detalhes, veja a página Hiperparâmetros.*

        - **Avaliar o módulo hidrológico isolado** — medir o NSE só do
          $Q_{physics}$ (saída do TTD+SCS, antes da LSTM corrigir).
          Isso mostra **quanto do desempenho vem da física pura** e
          quanto vem da correção neural.

        - **Comparação direta com EA-LSTM e δHBV** — garantir que
          estamos no estado da arte internacional.
        """
    )

with tab_fase2_med:
    st.markdown(
        """
        **Desenvolvimentos durante a Fase 2 (M9–M16):**

        - **Evolução do SCS-CN** — ele teve contribuição limitada na
          Fase 1 (bacia única). Três alternativas em investigação:
          - **ASMA** (Verma et al., 2020) — versão do SCS com contabilidade
            contínua de umidade antecedente
          - **Reservatório de produção do GR4J** — uma equação simples
            de armazenamento, já usada em modelos diferenciáveis
          - **Módulo de escoamento do MGB** (Collischonn et al., 2007) —
            baseado no conceito VIC, mantém memória de umidade entre
            passos de tempo

        - **$t_{c\\_scale}$ dinâmico** — hoje é um número fixo aprendido
          durante o treino. Na Fase 2 queremos que **varie com o
          estado da bacia**: em bacias secas, a água demora mais para
          chegar; em bacias saturadas, o escoamento é mais rápido.

        - **Separação explícita de fluxo de base** — hoje a LSTM captura
          implicitamente o fluxo de base (a contribuição lenta do
          aquífero). Vamos testar um **filtro diferenciável** que
          separa explicitamente as componentes rápida e lenta da
          vazão, com parâmetros interpretáveis.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Cronograma visual
# ---------------------------------------------------------------------------
st.subheader("Cronograma — próximos 20 meses")

tasks = [
    ("Fase 1: Estudo comparativo (bacia única)", "M1", "M6", "#16a34a"),
    ("Redação do Artigo 1", "M7", "M8", "#2563eb"),
    ("Coleta de dados (~100 bacias)", "M9", "M10", "#f59e0b"),
    ("Fase 2: Treinamento multi-bacia", "M11", "M14", "#9333ea"),
    ("Redação do Artigo 2", "M15", "M16", "#2563eb"),
    ("Redação da tese", "M17", "M19", "#dc2626"),
    ("Defesa final", "M20", "M20", "#000000"),
]


def _m(s):
    return int(s.replace("M", ""))


fig = go.Figure()
for name, m0, m1, color in tasks:
    fig.add_trace(go.Bar(
        y=[name], x=[_m(m1) - _m(m0) + 1],
        base=[_m(m0)],
        orientation="h",
        marker=dict(color=color),
        text=f"{m0} → {m1}",
        textposition="inside",
        showlegend=False,
    ))
fig.update_layout(
    template="plotly_white",
    xaxis=dict(title="Meses a partir da qualificação (abril/2026)",
               range=[0, 21], dtick=2),
    yaxis=dict(autorange="reversed"),
    height=380,
    title="Cronograma — próximos 20 meses",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Status: **Fase 1 concluída** ✅. Próxima etapa imediata: submissão "
    "do Artigo 1 (até ~junho/2026) e início da coleta multi-bacia."
)

st.divider()

# ---------------------------------------------------------------------------
# Publicações
# ---------------------------------------------------------------------------
st.subheader("Artigos planejados")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        """
        #### 📄 Artigo 1 — Metodologia (Fase 1)

        *A Differentiable Hydrological Framework Integrating
        Distributed Unit Hydrograph, SCS-CN, and LSTM for Hourly
        Streamflow Prediction*

        **O que conta:** apresentação do framework, estudo comparativo
        dos 10 modelos, caracterização do trade-off entre previsão e
        simulação, interpretação física dos parâmetros aprendidos.

        **Destino:** *Journal of Hydrology* ou *Water Resources Research*.
        **Submissão:** junho/2026 (M8).
        """
    )

with col_p2:
    st.markdown(
        """
        #### 📄 Artigo 2 — Regionalização (Fase 2)

        *Prediction in Ungauged Basins via Regionalized Hydrological
        Parameters: A Multi-Catchment Differentiable Framework for
        Brazil*

        **O que conta:** arquitetura com codificador, experimentos em
        ~100 bacias, comparação com EA-LSTM e δHBV, análise de
        transferibilidade por bioma brasileiro.

        **Destino:** *Journal of Hydrology* ou *HESS*.
        **Submissão:** fevereiro/2027 (M16).
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# O que vem depois — impacto esperado
# ---------------------------------------------------------------------------
st.subheader("Se der certo, o que a gente deixa?")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        **🔬 Científico**

        Uma **arquitetura modular** validada em bacias brasileiras —
        outros pesquisadores podem trocar módulos (ex.: SCS-CN → GR4J →
        MGB) sem refazer tudo.

        Código aberto, dados processados e modelos treinados
        disponíveis publicamente — reprodutibilidade total.
        """
    )

with c2:
    st.markdown(
        """
        **🏗️ Aplicado**

        Um sistema capaz de gerar **séries sintéticas de vazão para
        qualquer bacia brasileira**, sem exigir estação fluviométrica.

        Útil para: sistemas de alerta, outorga, dimensionamento de
        obras hidráulicas, análise de cenários de mudança de uso
        do solo, ANA, CEIVAP e comitês de bacia em geral.
        """
    )

with c3:
    st.markdown(
        """
        **🌎 Impacto social**

        A maioria das bacias brasileiras vulneráveis a cheias está
        justamente **onde o monitoramento é mais escasso** (Norte,
        Nordeste, periferias urbanas).

        Um modelo que funciona em bacias não monitoradas **democratiza**
        o acesso à informação hidrológica para regiões que mais
        precisam.
        """
    )

st.caption(
    "*O objetivo final do doutorado é esse: reduzir a distância entre "
    "onde a hidrologia sabe prever e onde a previsão é mais necessária.*"
)
