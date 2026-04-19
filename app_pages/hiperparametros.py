"""
Hiperparametros — configuracao da Fase 1 e nota metodologica sobre as
variacoes testadas.
"""
from __future__ import annotations

import streamlit as st

import data_loader as dl
from embed_mode import maybe_hide_chrome

maybe_hide_chrome()

st.title("Hiperparâmetros")
st.caption(
    "A configuração usada nos 10 modelos apresentados e o registro das "
    "variações testadas"
)

with st.expander("🤔 O que são *hiperparâmetros* e por que documentar?"):
    st.markdown(
        """
        Redes neurais têm dois tipos de "ajustes":

        - **Parâmetros** — os pesos internos da rede, aprendidos
          automaticamente durante o treinamento (milhares ou milhões
          deles)
        - **Hiperparâmetros** — as escolhas **estruturais** feitas
          antes de treinar: quantas camadas a rede tem, quantos
          neurônios em cada camada, quantas horas de histórico ela
          olha, etc.

        **Por que documentar?** Para reprodutibilidade e transparência
        metodológica. A configuração da Fase 1 foi definida a partir
        de experimentos preliminares e boas práticas consolidadas
        (Kratzert et al., 2019). Além disso, fizemos uma busca
        sistemática em grade para registrar **quais variações foram
        testadas** — mostrando que a escolha não foi arbitrária.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# 1. Configuração usada nos resultados apresentados
# ---------------------------------------------------------------------------
st.subheader("Configuração dos 10 modelos apresentados")

st.markdown(
    """
    Todos os 10 modelos do estudo comparativo (páginas *Ablação &
    Trade-off*, *Explorar hidrogramas*) foram treinados com **a mesma
    configuração de hiperparâmetros** — para garantir que as diferenças
    observadas decorram **apenas dos módulos físicos** (presença/ausência
    do TTD, SCS-CN, parâmetros fixos vs. aprendíveis), e não de
    variações arquiteturais.
    """
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        #### Arquitetura da LSTM

        | Hiperparâmetro | Valor |
        |---|---|
        | **Camada oculta** (*hidden size*) | 64 |
        | **Número de camadas** | 2 |
        | **Dropout** (Srivastava et al., 2014) | 0,1 |
        | **Janela de histórico** (*lookback*) | 240 h (10 dias) |
        | **Horizonte de previsão** | 24 h (multi-passo) |
        """
    )

with col_b:
    st.markdown(
        """
        #### Otimização

        | Hiperparâmetro | Valor |
        |---|---|
        | **Otimizador** | Adam |
        | **Taxa de aprendizado** (*lr*) | 1e-3 |
        | **Weight decay** (L2) | 1e-5 |
        | **Batch size** | 1.024 |
        | **Gradient clipping** | 1,0 |
        | **Épocas** (máx) | 300 |
        | **Early stopping** (paciência) | 30 |
        | **Semente aleatória** | 42 |
        """
    )

st.info(
    "**Função de perda:** MSE em escala logarítmica — "
    "$\\mathcal{L} = \\frac{1}{N \\cdot H} \\sum \\left[\\log(1 + \\hat{Q}) - "
    "\\log(1 + Q_{obs})\\right]^2$. Penaliza proporcionalmente erros em "
    "vazões baixas e altas, evitando o viés em direção a picos do MSE em "
    "escala linear. Detalhes em **Metodologia → LSTM**."
)

st.caption(
    "**Hardware:** NVIDIA RTX 3000 Ada (8 GB) · "
    "**Tempo médio por modelo:** ~1–2 h · "
    "**Data do treinamento:** janeiro/2026. "
    "Os arquivos `results.json` de cada modelo (em "
    "`outputs/final/ablation/`) contêm o número efetivo de épocas até "
    "o *early stopping*."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. Variações testadas (sem métricas)
# ---------------------------------------------------------------------------
st.subheader("Variações testadas")

st.markdown(
    """
    Para validar que a configuração acima é adequada — e para informar
    as escolhas arquiteturais da **Fase 2** (cenário multi-bacia) —
    conduzimos uma **busca em grade** explorando variações dos principais
    hiperparâmetros. O registro dessas variações está documentado
    abaixo como nota metodológica.
    """
)

col_var, col_fix = st.columns(2)

with col_var:
    st.markdown(
        """
        #### O que foi variado

        | Hiperparâmetro | Valores testados |
        |---|---|
        | **Janela de histórico** | 72 · 120 · 168 · **240** h |
        | **Camada oculta** (*hidden size*) | 32 · **64** · 128 |
        | **Número de camadas LSTM** | 1 · **2** · 3 |
        | **Tipo de modelo** | 2 variantes do TTD-SCS-LSTM |

        *Em **negrito**, os valores que ficaram na Fase 1.*
        """
    )

with col_fix:
    st.markdown(
        """
        #### O que foi mantido fixo

        | Hiperparâmetro | Valor |
        |---|---|
        | Horizonte de previsão | 24 h |
        | *Dropout* | 0,1 |
        | Taxa de aprendizado | 1e-3 |
        | *Weight decay* | 1e-5 |
        | *Batch size* | 1.024 |
        | Otimizador | Adam |
        | Semente aleatória | 42 |
        """
    )

# Total de combinações
try:
    df = dl.load_hyperparam_search()
    n_configs = len(df) if not df.empty else 108
except Exception:
    n_configs = 108

st.markdown(
    f"""
    **Total de configurações treinadas na busca:** {n_configs}. Cada
    uma foi treinada individualmente com a mesma bacia (Rio Preto),
    mesma semente e treinamento reduzido (150 épocas, paciência 20)
    para viabilizar o custo computacional.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# 3. O que indica pra Fase 2
# ---------------------------------------------------------------------------
st.subheader("O que isso indica para a Fase 2")

st.markdown(
    """
    Três sinais extraídos da busca orientam a continuação do trabalho:

    - **A janela de 240 h da Fase 1 é adequada** — coerente com o
      tempo de viagem máximo da bacia e com a memória de recessão
    - **64 neurônios é o ponto de equilíbrio na bacia única** — com
      mais dados (multi-bacia), arquiteturas mais largas (128) podem
      ser vantajosas
    - **3 camadas LSTM** tendem a superar 2 quando há mais variabilidade
      nos dados — vale reavaliar na Fase 2
    """
)

st.success(
    "**Para a Fase 2:** planejamos uma busca mais refinada no cenário "
    "multi-bacia, onde o aumento do volume de dados pode tornar "
    "arquiteturas mais profundas (3 camadas, 128 neurônios) mais "
    "vantajosas. Em vez de *grid search* (testar todas as combinações "
    "possíveis, como fizemos agora), pretendemos usar **Optuna**."
)

with st.expander("🔎 O que é Optuna?"):
    st.markdown(
        """
        **Optuna** é uma biblioteca de código aberto para **otimização
        automatizada de hiperparâmetros**. A diferença para o *grid
        search* tradicional é como a busca decide quais combinações
        testar:

        | | *Grid search* (o que fizemos) | **Optuna** (planejado) |
        |---|---|---|
        | **Estratégia** | Testa **todas** as combinações possíveis | **Aprende com as rodadas anteriores** para escolher a próxima |
        | **Custo** | Cresce multiplicativamente (108 configs para 3 parâmetros) | Converge para boas configurações com muito menos tentativas |
        | **Quando usar** | Espaço de busca pequeno, como na Fase 1 | Espaço de busca maior, como na Fase 2 (multi-bacia) |

        Tecnicamente, o Optuna usa um algoritmo chamado **TPE**
        (*Tree-structured Parzen Estimator*) — uma forma de **otimização
        bayesiana**. A ideia: ele mantém um modelo probabilístico de
        "quais regiões do espaço de hiperparâmetros tendem a produzir
        bons resultados" e **prioriza testar lá**. A cada rodada o
        modelo se atualiza. É como se fosse um caçador que, em vez de
        vasculhar o mato inteiro, usa pegadas e rastros para decidir
        onde procurar em seguida.

        **Por que isso importa na Fase 2?** Com ~100 bacias, o custo
        computacional de cada treinamento aumenta muito. Uma busca
        exaustiva seria impraticável. O Optuna permite **gastar o
        mesmo orçamento computacional de forma mais inteligente** —
        idealmente encontrando configurações melhores com menos
        treinamentos.

        *Documentação:* [optuna.org](https://optuna.org/)
        """
    )
