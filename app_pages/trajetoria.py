"""
Trajetoria da pesquisa — da arquitetura exploratoria (Tensor Hydro) ao
TTD-SCS-LSTM.

Secao extraida da introducao.tex (Trajetoria da Pesquisa).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from embed_mode import maybe_hide_chrome

maybe_hide_chrome()

st.title("Trajetória da pesquisa")
st.caption(
    "Do modelo exploratório Tensor Hydro ao TTD-SCS-LSTM — "
    "uma simplificação deliberada por fundamentação hidrológica"
)

# ---------------------------------------------------------------------------
# Tese de abertura
# ---------------------------------------------------------------------------
st.markdown(
    """
    A arquitetura atual **não foi a primeira tentativa**. Desde o início
    do doutorado, em **março de 2024**, a pesquisa passou por um ciclo
    de experimentação → diagnóstico → simplificação. A trajetória importa
    porque é o próprio argumento do trabalho: **a física certa vale mais
    que a rede neural mais elaborada**.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Fase 0 — Tensor Hydro
# ---------------------------------------------------------------------------
st.subheader("Fase 0 — Tensor Hydro (arquitetura exploratória)")

col_arch, col_stats = st.columns([0.6, 0.4])

with col_arch:
    st.markdown(
        """
        Previsão **diária** de vazão em toda uma **rede de drenagem
        distribuída**, com múltiplos pontos de previsão simultâneos.

        **Arquitetura (resumida, em termos simples):**
        - **PatchCNN** — uma rede neural que "olhava" recortes de mapa
          (32×32 pixels) para extrair "impressões digitais" de cada
          pedaço da bacia: elevação, declividade, sentido do escoamento,
          uso do solo, curvatura do terreno
        - **FiLM** (*Feature-wise Linear Modulation*) — mecanismo que
          ajusta a rede com informação do local específico
        - **Módulos temporais intercambiáveis** — testamos Transformer,
          LSTM, TCN e Mamba como "motor" de processamento das séries
          temporais de chuva
        - Bacia dividida em **689 ottobacias**
        - Escala **diária**, "memória" de **365 dias** (a rede olhava
          para 1 ano passado para prever o dia seguinte)
        - Testado com e sem vazão observada como entrada adicional
        """
    )

with col_stats:
    st.markdown("**Resultados na estação monitorada (treino+validação):**")
    st.metric(
        "NSE — sem vazão observada", "0,72",
        help="Só chuva como entrada (P → Q) — configuração compatível com PUB",
    )
    st.metric(
        "NSE — com vazão observada passada", "0,79",
        help="Chuva + vazão medida nas horas/dias anteriores (P + Q_obs → Q). "
             "Facilita a previsão, mas exige estação telemétrica.",
    )
    st.caption(
        "Resultados promissores *localmente*. O problema apareceu na "
        "avaliação espacial."
    )

# Visualização conceitual da degradação
st.markdown("#### O problema: degradação em *leave-one-station-out*")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Estação de treinamento", "Leave-one-station-out", "Pontos sem monitoramento"],
    y=[0.79, 0.40, 0.20],  # valores ilustrativos
    marker=dict(color=["#16a34a", "#f59e0b", "#dc2626"]),
    text=[f"NSE ≈ {v:.2f}" for v in [0.79, 0.40, 0.20]],
    textposition="outside",
))
fig.add_hline(y=0.75, line_dash="dash", line_color="gray",
              annotation_text="NSE Muito Bom")
fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
              annotation_text="NSE Aceitável")
fig.update_layout(
    title="Desempenho do Tensor Hydro por cenário de validação (ilustrativo)",
    template="plotly_white",
    yaxis=dict(title="NSE", range=[0, 1]),
    height=400,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    O modelo havia **aprendido padrões estatísticos da estação onde
    foi treinado**, mas não tinha embutido as equações que governam o
    processo físico (tempo que a água leva para chegar ao exutório, o
    que o solo absorve antes de começar a escorrer, como cada pedaço
    da bacia contribui na proporção da sua área). Não havia
    **estrutura física transferível** para locais novos.

    *A boa métrica local mascarava a incapacidade de generalizar: o
    modelo não sabia hidrologia, sabia replicar a relação chuva-vazão
    daquele ponto específico.*

    Somava-se a isso a **complexidade arquitetural** — vários módulos
    neurais interdependentes, que dificultavam identificar *onde*
    estava o erro: era o PatchCNN? o FiLM? o módulo temporal?
    """
)

st.divider()

# ---------------------------------------------------------------------------
# A pivotagem
# ---------------------------------------------------------------------------
st.subheader("Três pivotagens convergentes")

piv1, piv2, piv3 = st.columns(3)

with piv1:
    st.markdown(
        """
        #### 1. Simplificação arquitetural com física

        Substituir módulos **neurais** por **camadas diferenciáveis**
        que implementam equações hidrológicas conhecidas:

        - PatchCNN para *embeddings* geomorfológicos → **TTD baseado
          em Maidment** (campo de velocidades a partir do DEM)
        - Módulos neurais de particionamento → **SCS-CN diferenciável**
          com $\\lambda$ aprendível

        **Resultado:** espaço de parâmetros muito menor, estrutura
        física interpretável e transferível.
        """
    )

with piv2:
    st.markdown(
        """
        #### 2. Mudança de escala temporal

        **Diário → horário** (horizontes $T{+}1$ até $T{+}24$):

        - Operação de reservatórios em cascata demanda
          resolução horária
        - Protocolos de emergência para inundação exigem
          antecedência de horas
        - Eventos convectivos tropicais se desenvolvem em escala
          sub-diária — invisíveis na escala diária
        - Curvas de permanência a partir de simulação contínua
          para outorga
        """
    )

with piv3:
    st.markdown(
        """
        #### 3. Prever a vazão a partir da chuva, apenas

        **Apenas** precipitação e atributos da bacia como entrada
        ($P \\rightarrow Q$) — **sem usar vazão observada**:

        - **Física correta**: a chuva é a *causa*, a vazão é a
          *consequência* — e não o contrário
        - **Viável em tempo real** sem depender de telemetria
          fluviométrica
        - **Aplicável a bacias sem monitoramento** — onde a vazão
          observada simplesmente não existe

        Foi o **único caminho compatível com o problema das bacias
        não monitoradas**.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
st.subheader("Resultado: o contraste")

col_t, col_t2 = st.columns(2)

with col_t:
    st.markdown("#### Tensor Hydro (fase exploratória)")
    st.markdown(
        """
        - PatchCNN + FiLM + Transformer
        - 689 ottobacias, *rasters* 32×32 px, 6 bandas
        - Milhões de parâmetros neurais
        - Escala diária, *lookback* 365 dias
        - **NSE 0,72 não autorregressivo / 0,79 autorregressivo**
        - **Degradou em leave-one-station-out**
        """
    )

with col_t2:
    st.markdown("#### TTD-SCS-LSTM (Fase 1)")
    st.markdown(
        """
        - TTD (Maidment gaussiano) + SCS-CN + LSTM
        - 245 ottobacias, atributos agregados
        - ~120 mil parâmetros (a maioria na LSTM)
        - **Escala horária**, *lookback* 240 h
        - **NSE 0,84** previsão 6h / **0,82** simulação contínua
        - Parâmetros físicos **interpretáveis e potencialmente
          regionalizáveis**
        """
    )

st.success(
    "**A tese:** o modelo TTD-SCS-LSTM, com arquitetura mais simples e "
    "fundamentada em equações hidrológicas consolidadas, superou o Tensor "
    "Hydro com **resolução temporal superior** e **parâmetros físicos "
    "interpretáveis**. A trajetória não é abandono de complexidade por "
    "limitação técnica — é a **substituição deliberada de complexidade "
    "arquitetural por fundamentação hidrológica**."
)

st.divider()

# ---------------------------------------------------------------------------
# Linha do tempo
# ---------------------------------------------------------------------------
st.subheader("Linha do tempo")

timeline = [
    ("🎓 2024 · mar",
     "**Início do doutorado.** Cumprimento de créditos e disciplinas "
     "do programa. Formulação do problema PUB como foco central da "
     "pesquisa."),
    ("📚 2024 · 2º sem",
     "Conclusão dos créditos e aprofundamento na literatura de "
     "modelagem hidrológica com aprendizado profundo."),
    ("🧪 2025 · 1º sem",
     "**Tensor Hydro v1** — primeira arquitetura: PatchCNN + LSTM em "
     "escala diária, com 689 ottobacias."),
    ("🧪 2025 · 2º sem",
     "**Tensor Hydro v2** — adição de FiLM e encoders alternativos "
     "(Transformer, Mamba). **Avaliação leave-one-station-out revelou "
     "a degradação** fora da estação de treino."),
    ("🔄 2025 · fim/2026 · jan",
     "**Pivot metodológico:** simplificação arquitetural, substituição "
     "de módulos neurais por camadas físicas diferenciáveis, mudança "
     "de escala (diária → horária) e de modo (autorregressivo → "
     "não autorregressivo)."),
    ("⚙️ 2026 · jan",
     "Implementação e treinamento dos 10 modelos do **TTD-SCS-LSTM** "
     "(300 épocas, SEED=42). Resultados finais: **NSE 0,84 em "
     "previsão 6 h · NSE 0,82 em simulação contínua**."),
    ("🎯 **2026 · 22/abr**",
     "**Defesa da qualificação** (você está aqui)."),
    ("🚀 2026 · mai–jun",
     "**Fase 2 — Regionalização multi-bacia.** Treinamento em ~100 "
     "bacias brasileiras + validação *leave-one-basin-out*. "
     "Meta: primeiros resultados até o **meio de 2026**."),
    ("📄 2026 · jun",
     "Submissão do **Artigo 1** (metodologia e estudo comparativo)."),
    ("📄 2027 · fev",
     "Submissão do **Artigo 2** (regionalização)."),
    ("🎓 2027 · fim / 2028 · 1º sem",
     "Redação e defesa final da tese."),
]

for data, evento in timeline:
    c1, c2 = st.columns([0.22, 0.78])
    c1.markdown(data)
    c2.markdown(evento)

st.caption(
    "*Os anos iniciais incluem o período de créditos (2024) antes da "
    "pesquisa técnica começar. As datas futuras (pós-defesa) são "
    "planejamento do cronograma detalhado na página **Fase 2**.*"
)
