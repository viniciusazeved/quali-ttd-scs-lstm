"""
Pagina de metodologia — arquitetura TTD-SCS-LSTM diferenciavel.

Conteudo fiel ao Cap. 3 da V31_Final da qualificacao.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from plots import plot_iuh

st.title("Metodologia — arquitetura hidrológica diferenciável")
st.caption(
    "Modelo TTD-SCS-LSTM: três módulos integrados em uma arquitetura treinada "
    "end-to-end via backpropagation"
)

# ---------------------------------------------------------------------------
# Visão geral + premissa
# ---------------------------------------------------------------------------
st.markdown(
    r"""
    A arquitetura segue o paradigma **sequencial**, em que a precipitação
    é processada por módulos hidrológicos diferenciáveis e a saída é
    refinada por uma rede neural. O fluxo de dados é:
    """
)

st.latex(r"P \xrightarrow{\;\text{SCS}\;} P_e \xrightarrow{\;\text{TTD}\;} Q_{\text{physics}} \xrightarrow{\;\text{LSTM}\;} Q_{\text{pred}}")

st.markdown(
    r"""
    em que $P_e$ é a precipitação efetiva (após separação pelo SCS-CN),
    $Q_{\text{physics}}$ é a vazão estimada pela convolução TTD, e
    $Q_{\text{pred}}$ é a saída final da LSTM, que recebe $Q_{\text{physics}}$,
    $P$ e variáveis temporais como entrada.
    """
)

st.info(
    "**Premissa fundamental:** o modelo é **não autorregressivo** — prevê $Q$ "
    "usando apenas $P$ e atributos fisiográficos, sem vazão observada como "
    "entrada. Isso é essencial para aplicação em **bacias não monitoradas** "
    "(PUB — *Prediction in Ungauged Basins*), já que a vazão observada em "
    "tempo real simplesmente não existe nessas bacias."
)

st.divider()

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
tab_diff, tab_scs, tab_ttd, tab_lstm, tab_treino, tab_abl = st.tabs([
    "Paradigma diferenciável", "SCS-CN", "TTD", "LSTM",
    "Treinamento & Loss", "Estudo comparativo & Hipóteses",
])

# ----------------------------------------------------------------- Paradigma
with tab_diff:
    st.subheader("O que torna este modelo *diferenciável*?")

    st.markdown(
        r"""
        Todos os módulos — inclusive os hidrológicos (SCS-CN e TTD) — são
        implementados como **camadas cujos gradientes podem ser calculados
        via *backpropagation***. Isso permite que o erro de previsão ajuste
        diretamente **tanto os pesos da rede neural** ($W$, $b$) **quanto
        os parâmetros físicos** ($\lambda$, $\sigma$, $t_{c\_scale}$).

        Esta propriedade é o que permite o treinamento **end-to-end**:
        todos os parâmetros da arquitetura são otimizados **simultaneamente**,
        e não em etapas separadas (calibrar física primeiro, depois treinar
        a LSTM).
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            r"""
            **Como isso é feito na prática?**

            1. **Substituir descontinuidades** por aproximações suaves.
               Ex.: a condição $P > \lambda S$ do SCS-CN clássico é
               substituída por $\mathrm{ReLU}(P - \lambda S)$, que tem
               subgradiente definido em todo o domínio.
            2. **Restringir parâmetros** a intervalos físicos via funções
               diferenciáveis (sigmoide, softplus, exp). Ex.:
               $\lambda = 0{,}01 + 0{,}39\,\sigma(\theta_\lambda)$.
            3. **Parametrizar o IUH** por uma distribuição gaussiana com
               parâmetros log-escalados, garantindo positividade e
               gradientes bem-comportados.
            """
        )

    with col_b:
        st.markdown(
            r"""
            **Por que não é uma PINN?**

            Este modelo **não é uma Physics-Informed Neural Network**
            (Raissi et al., 2019). PINNs aprendem soluções de equações
            diferenciais parciais (Saint-Venant, Navier-Stokes) diretamente
            pela rede neural, com a física entrando na função de perda.

            Aqui, as equações (SCS-CN, hidrograma unitário) são
            **parametrizações conceituais consolidadas da hidrologia
            operacional** — não leis fundamentais. Sua reformulação como
            **camadas diferenciáveis** permite que codifiquem décadas de
            conhecimento hidrológico como *conhecimento prévio estrutural*
            que guia o aprendizado.
            """
        )

    st.markdown(
        """
        #### Posicionamento no paradigma diferenciável

        Segundo a taxonomia de Shen et al. (2023), este trabalho
        combina três estratégias:

        - **Substituição de submódulos conceituais** por camadas
          diferenciáveis com parâmetros aprendíveis (SCS, TTD)
        - **Correção de erros residuais** via rede neural (LSTM)
        - **Incorporação de restrições físicas** como regularização
          estrutural (não autorregressivo, unidades consistentes,
          conservação de massa no IUH)
        """
    )

    st.caption(
        "Referências-chave: Shen et al. (2023) — *Differentiable modeling to "
        "augment machine learning for scientific discovery*; Nearing et al. "
        "(2021) — *What role does hydrological science play in the age of "
        "machine learning?*; Feng et al. (2022) — $\\delta$HBV."
    )

# ---------------------------------------------------------------------- SCS
with tab_scs:
    st.subheader("Módulo SCS-CN — separação de escoamento")

    st.markdown(
        r"""
        O método SCS-CN separa a chuva total ($P$) em **precipitação
        efetiva** ($P_e$) — parcela que gera escoamento superficial — em
        função do Curve Number ($CN$) e do coeficiente de abstração inicial
        $\lambda$.

        **Formulação clássica:**
        """
    )

    st.latex(r"S = 25{,}4 \left(\frac{1000}{CN} - 10\right) \quad \text{(mm)}")
    st.latex(r"P_e = \begin{cases} 0, & P \leq \lambda S \\ \dfrac{(P - \lambda S)^2}{P - \lambda S + S}, & P > \lambda S \end{cases}")

    st.markdown(
        r"""
        **Implementação como camada diferenciável** — a condição discreta
        $P > \lambda S$ é substituída por $\mathrm{ReLU}(\cdot)$, que tem
        subgradiente definido em todo o domínio:
        """
    )

    st.latex(r"P_e = \frac{\mathrm{ReLU}(P - \lambda S)^2}{\mathrm{ReLU}(P - \lambda S) + S + \epsilon}")

    st.markdown(
        r"""
        com $\epsilon = 10^{-6}$ para estabilidade numérica.

        **Parâmetro aprendível** — $\lambda$ é restrito ao intervalo
        $[0{,}01;\,0{,}40]$ através de uma transformação sigmoide:
        """
    )

    st.latex(r"\lambda = 0{,}01 + 0{,}39\;\sigma(\theta_\lambda)")

    st.markdown(
        r"""
        em que $\theta_\lambda$ é o parâmetro interno otimizado via
        *backpropagation*. A escolha do intervalo tem fundamento empírico:
        Woodward et al. (2003) demonstraram $\lambda \approx 0{,}05$ para
        a maioria das bacias analisadas; ValleJunior et al. (2019)
        reportaram mediana de **0,045** em bacias tropicais brasileiras,
        com 96,7% dos valores inferiores a 0,2 — o valor clássico
        **superestima sistematicamente** a abstração inicial em condições
        tropicais.

        **Por que manter $CN$ fixo?** Os valores de $CN$ são do produto
        oficial BHAE_CN-2022 da ANA (baseado em Sartori et al. 2005, com
        classificação de solos adaptada para Brasil + MapBiomas Col. 8.0).
        Manter $CN$ fixo garante **consistência com o produto oficial** e
        **viabiliza a transferência direta** para bacias não monitoradas
        na Fase 2 (regionalização).
        """
    )

    st.markdown("#### Simulador interativo — SCS-CN")
    col1, col2, col3 = st.columns(3)
    with col1:
        cn = st.slider("CN", 30, 100, 63, 1,
                       help="Bacia do Rio Preto: CN médio = 63,0")
    with col2:
        lam = st.slider("λ", 0.01, 0.40, 0.20, 0.01,
                        help="Clássico 0.2; tropicais brasileiras ~0.045")
    with col3:
        p_max = st.slider("P máx (mm)", 10, 200, 80, 10)

    P = np.linspace(0, p_max, 200)
    S = 25.4 * (1000 / cn - 10)
    relu_arg = np.maximum(P - lam * S, 0)
    Pe = relu_arg ** 2 / (relu_arg + S + 1e-6)

    fig_scs = go.Figure()
    fig_scs.add_trace(go.Scatter(
        x=P, y=Pe, mode="lines",
        line=dict(color="#2563eb", width=3),
        name="P_e (escoamento efetivo)",
    ))
    fig_scs.add_trace(go.Scatter(
        x=P, y=P, mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Q = P (100% escoamento)",
    ))
    fig_scs.add_vline(x=lam * S, line_dash="dot", line_color="red",
                       annotation_text=f"λS = {lam * S:.1f} mm")
    fig_scs.update_layout(
        title=f"CN = {cn}, λ = {lam:.2f} → S = {S:.1f} mm",
        template="plotly_white",
        xaxis_title="Precipitação P (mm)",
        yaxis_title="Precipitação efetiva P_e (mm)",
        height=400,
    )
    st.plotly_chart(fig_scs, use_container_width=True)

# ---------------------------------------------------------------------- TTD
with tab_ttd:
    st.subheader("Módulo TTD — Travel Time Distribution")

    st.markdown(
        r"""
        O TTD propaga o escoamento de cada ottobacia até o exutório via
        **convolução com o hidrograma unitário instantâneo (IUH)**. Cada
        uma das 245 ottobacias tem seu próprio IUH, parametrizado pelo
        respectivo tempo de concentração $T_c$ — o conjunto das 245
        respostas individuais constitui um **hidrograma unitário
        espacialmente distribuído**:
        """
    )

    st.latex(r"Q_{\text{physics}}(t) = \sum_{i=1}^{245} \left[ P_e^{(i)} * h^{(i)} \right] \cdot \frac{A_i}{A_{\text{total}}} \cdot k")

    st.markdown(
        r"""
        em que $P_e^{(i)}$ é a precipitação efetiva na ottobacia $i$,
        $h^{(i)}$ é o IUH da ottobacia $i$, $A_i$ é sua área e $k$ é o
        fator de conversão de mm/h para m³/s.
        """
    )

    st.markdown("#### Cálculo do $T_c$ — método de Maidment (1996)")

    st.markdown(
        r"""
        O tempo de concentração é derivado de um **campo de velocidades
        espacialmente distribuído**, com velocidade em cada pixel função
        da declividade local e da área de drenagem acumulada:
        """
    )

    st.latex(r"V = V_m \cdot \frac{S^b \cdot A^c}{\overline{S^b \cdot A^c}}")

    st.markdown(
        r"""
        com $b = c = 0{,}5$ (valores típicos reportados em Maidment, 1996).
        O tempo de viagem de cada pixel é integrado ao longo do caminho
        de fluxo via algoritmo D8 (WhiteboxTools):
        """
    )

    st.latex(r"T_c(i) = \sum_{j \in \text{caminho}(i)} \frac{L_j}{V_j}")

    st.markdown(
        r"""
        **Ajuste pelo coeficiente de Manning** — o $T_c$ base pode ser
        refinado pela rugosidade da superfície (derivada do LULC):
        """
    )

    st.latex(r"T_{c,\text{Manning}} = T_{c,\text{base}} \cdot \frac{n_{\text{local}}}{n_{\text{ref}}}")

    st.markdown(
        r"""
        com $n_{\text{ref}} = 0{,}035$ (agropecuária). Exemplos:
        floresta $n = 0{,}15$ ($\times 4{,}3$), urbano $n = 0{,}015$
        ($\times 0{,}43$). Na bacia do rio Preto, o Manning eleva o $T_c$
        médio de **9,5 h → 23,4 h** (+145%), refletindo a predominância
        de cabeceiras florestadas na Serra da Mantiqueira.
        """
    )

    st.markdown("#### IUH parametrizado como gaussiano diferenciável")

    st.latex(r"h(t) = \frac{A_{\text{total}}}{\sigma\sqrt{2\pi}} \exp\!\left[-\frac{(t - T_c)^2}{2\sigma^2}\right]")

    st.markdown(
        r"""
        **Parâmetros aprendíveis** (ambos em escala logarítmica para
        garantir positividade):

        - $T_c^{\text{efetivo}} = T_c^{\text{base}} \cdot \exp(\theta_{tc})$
          — inicializado em $\theta_{tc} = 0 \Rightarrow t_{c\_scale} = 1$
        - $\sigma = \exp(\theta_\sigma)$ — inicializado em $\sigma_0 = 3{,}0$ h

        O $T_c$ calculado (Maidment + Manning) serve como ***prior* físico
        informativo**, não como valor prescritivo. O modelo ajusta
        $t_{c\_scale}$ durante o treinamento para encontrar o tempo de
        resposta efetivo da bacia, corrigindo eventuais superestimativas
        da formulação empírica nas cabeceiras florestadas.
        """
    )

    st.caption(
        "**Por que gaussiano e não gamma?** (1) Parcimônia equivalente "
        "(2 parâmetros em ambas); (2) estabilidade numérica — gamma com "
        "n < 1 diverge em t = 0; (3) a LSTM compensa a simetria residual. "
        "A substituição por gamma ou IUH discreto é aprimoramento "
        "planejado para a Fase 2."
    )

    st.markdown("#### Simulador — IUH Gaussiano")
    col_a, col_b = st.columns(2)
    with col_a:
        tc_values = st.multiselect(
            "Tempos de concentração (h)",
            options=[0.5, 1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 76],
            default=[3, 12, 24, 48],
            help="Na bacia do Rio Preto: Tc médio (Manning) = 23,4 h; máx = 76,6 h",
        )
    with col_b:
        sigma = st.slider("σ (h)", 0.5, 12.0, 3.0, 0.5,
                          help="Valor inicial no treinamento: 3,0 h")

    if tc_values:
        st.plotly_chart(plot_iuh(tc_values, sigma=sigma, duration=96),
                         use_container_width=True)

# ---------------------------------------------------------------------- LSTM
with tab_lstm:
    st.subheader("Módulo neural — LSTM como refinamento")

    st.markdown(
        r"""
        A LSTM (Hochreiter & Schmidhuber, 1997) **não é o modelo** — é o
        **componente de refinamento** que corrige a estimativa física e
        captura padrões não representados pelos módulos simplificados:

        - Umidade antecedente não capturada pelo CN estático
        - Não-linearidades residuais na relação chuva-vazão
        - Dinâmica de fluxo de base (hipótese a investigar na Fase 2)
        """
    )

    st.markdown(
        """
        #### Hiperparâmetros (definidos após busca preliminar)

        | Parâmetro | Valor |
        |---|---|
        | Hidden size | 64 |
        | Número de camadas | 2 |
        | Dropout (Srivastava et al., 2014) | 0,1 |
        | *Lookback* | **240 h (10 dias)** |
        | *Horizon* | 24 h (multi-passo) |
        | Otimizador | Adam (lr = 1e-3, weight_decay = 1e-5) |
        | *Batch size* | 1.024 |
        | GPU | NVIDIA RTX 3000 Ada (8 GB) |
        """
    )

    st.markdown(
        r"""
        **Justificativa do *lookback* de 240 h:** considera duas escalas
        temporais complementares. O tempo de viagem máximo no *raster*
        ($T_c^{\max} \approx 105$ h) define o horizonte necessário para
        contemplar a propagação desde o pixel mais distante. Além disso,
        a cauda do IUH se estende além do $T_c$ máximo — evidenciando a
        dinâmica de recessão que a LSTM captura implicitamente como
        fluxo de base. A margem adicional (~135 h) permite à rede
        aprender essa dinâmica lenta a partir do histórico de
        precipitação acumulada.
        """
    )

    st.markdown(
        """
        #### Variáveis de entrada

        Três grupos:

        1. **Precipitação processada** — $\\log(1 + P)$, ou $\\log(1 + P_e)$
           quando o SCS está ativo
        2. **Variáveis temporais** — hora do dia e mês do ano, ambos
           normalizados em [0, 1]
        3. **Opcional: atributos físicos** — $CN/100$ e $T_c/24$ por
           ottobacia (quando usados como *features* estáticas)

        A distinção entre usar $CN$ e $T_c$ como *features* (a rede
        interpreta livremente) e implementá-los como parâmetros das
        camadas diferenciáveis (governam equações hidrológicas) é
        **central na arquitetura**: no segundo caso, gradientes fluem
        durante o treinamento.
        """
    )

    st.markdown(
        r"""
        #### Por que transformação $\log(1 + x)$?

        1. **Adequação à distribuição:** vazões e precipitações seguem
           distribuições log-normais/gama com cauda longa. A log
           "comprime" a cauda, aproximando de uma normal.
        2. **Estabilidade numérica:** o $+1$ garante que zero
           (precipitação nula) vire $\log(1) = 0$, evitando $\log(0)$.
        3. **Interpretabilidade + transferência:** preserva ordenação
           relativa sem exigir guardar estatísticas (média, desvio)
           do treino para aplicação em novas bacias.
        """
    )

# ----------------------------------------------------- Treinamento & Loss
with tab_treino:
    st.subheader("Treinamento end-to-end")

    st.markdown(
        r"""
        #### Saída — MLP decoder multi-horizonte
        """
    )
    st.latex(r"\hat{Q}_{t+1:t+24} = \mathrm{MLP}\bigl(\mathrm{LSTM}(x_{t-240:t})\bigr)")

    st.markdown(
        r"""
        #### Função de perda em escala logarítmica

        Penaliza proporcionalmente erros em vazões baixas e altas,
        evitando o viés em direção a picos característico do MSE em
        escala linear:
        """
    )
    st.latex(
        r"\mathcal{L} = \frac{1}{N \cdot H}\sum_{i=1}^{N}\sum_{h=1}^{H}"
        r"\left[\log(1 + \hat{Q}_h^{(i)}) - \log(1 + Q_h^{(i)})\right]^2"
    )

    st.markdown(
        """
        com $N$ = tamanho do *batch* e $H = 24$ = horizonte máximo de
        previsão. Esta escolha é consistente com a transformação das
        variáveis de entrada: **tudo em escala log, do começo ao fim**.
        """
    )

    st.divider()

    st.markdown("#### Dois modos de avaliação — o que significa cada um?")

    st.markdown(
        """
        O mesmo modelo é avaliado de **duas formas complementares**,
        que representam **dois produtos operacionais distintos** da
        hidrologia:
        """
    )

    # --- Diagrama 1: Forecasting multi-horizonte (janela deslizante)
    st.markdown("##### 1. Previsão multi-horizonte — *forecasting* (janela deslizante)")

    fig_fc = go.Figure()

    # Eixo temporal de 0 a 500h (ilustrativo)
    # Janela 1: input 0-240, output 240-264
    # Janela 2: input 24-264, output 264-288
    # Janela 3: input 48-288, output 288-312
    windows = [
        (0, 240, 240, 264, "Janela 1"),
        (24, 264, 264, 288, "Janela 2"),
        (48, 288, 288, 312, "Janela 3"),
    ]
    y_positions = [3, 2, 1]
    for (inp_s, inp_e, out_s, out_e, label), y in zip(windows, y_positions):
        fig_fc.add_trace(go.Scatter(
            x=[inp_s, inp_e, inp_e, inp_s, inp_s],
            y=[y - 0.35, y - 0.35, y + 0.35, y + 0.35, y - 0.35],
            fill="toself", fillcolor="rgba(37, 99, 235, 0.25)",
            line=dict(color="#2563eb", width=1),
            mode="lines",
            showlegend=(y == 3), name="Input P (240 h)",
            hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=[out_s, out_e, out_e, out_s, out_s],
            y=[y - 0.35, y - 0.35, y + 0.35, y + 0.35, y - 0.35],
            fill="toself", fillcolor="rgba(220, 38, 38, 0.35)",
            line=dict(color="#dc2626", width=1),
            mode="lines",
            showlegend=(y == 3), name="Output Q (24 h)",
            hoverinfo="skip",
        ))
        fig_fc.add_annotation(x=-5, y=y, text=label, showarrow=False,
                              xanchor="right", font=dict(size=12))
    # Marcadores dos horizontes avaliados
    for h_idx, h_label in enumerate(["1h", "3h", "6h", "12h", "24h"]):
        x_pos = 240 + [1, 3, 6, 12, 24][h_idx]
        fig_fc.add_annotation(x=x_pos, y=3.55, text=f"+{h_label}",
                              showarrow=True, arrowhead=2, arrowsize=0.8,
                              ax=0, ay=-25, font=dict(size=10, color="#7c2d12"))

    fig_fc.update_layout(
        template="plotly_white",
        xaxis=dict(title="Horas", range=[-20, 330]),
        yaxis=dict(range=[0.3, 4.2], showticklabels=False, showgrid=False),
        height=280,
        margin=dict(l=60, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown(
        """
        **Como funciona:**
        - A cada passo, o modelo recebe **240 h de precipitação passada** como entrada
        - Produz **24 h de previsão de vazão à frente** (vetor $\\hat{Q}_{t+1:t+24}$)
        - A janela desliza **1 hora por vez** → gera milhares de previsões independentes
        - Cada predição é **avaliada separadamente nos 5 horizontes**: 1h, 3h, 6h, 12h, 24h

        **Interpretação da métrica:**
        - **NSE em t+1h**: "Quão bem o modelo prevê a vazão 1 hora à frente?"
        - **NSE em t+6h**: "Quão bem o modelo prevê 6 horas à frente?" (**o melhor** no nosso caso: 0,84)
        - **NSE em t+24h**: "Quão bem prevê 24h à frente?" (pior, 0,75 — mais incerteza)

        **Aplicação operacional:** sistemas de alerta hidrológico, defesa civil,
        operação de reservatórios em tempo quase-real.
        """
    )

    st.divider()

    # --- Diagrama 2: Simulação contínua
    st.markdown("##### 2. Simulação contínua — *continuous simulation* (roda sem reset)")

    fig_cs = go.Figure()

    # Warmup 0-480, simulação 480-6600 (~9 meses)
    fig_cs.add_trace(go.Scatter(
        x=[0, 480, 480, 0, 0], y=[1, 1, 2, 2, 1],
        fill="toself", fillcolor="rgba(148, 163, 184, 0.4)",
        line=dict(color="#64748b", width=1), mode="lines",
        name="Warmup (480 h ≈ 20 dias)", hoverinfo="skip",
    ))
    fig_cs.add_trace(go.Scatter(
        x=[480, 6600, 6600, 480, 480], y=[1, 1, 2, 2, 1],
        fill="toself", fillcolor="rgba(37, 99, 235, 0.3)",
        line=dict(color="#2563eb", width=1.5), mode="lines",
        name="Simulação contínua (9 meses sem vazão observada)",
        hoverinfo="skip",
    ))
    # Anotações
    fig_cs.add_annotation(x=240, y=1.5, text="Warmup<br>inicializa estado<br>da LSTM",
                          showarrow=False, font=dict(size=11))
    fig_cs.add_annotation(x=3540, y=1.5,
                          text="O modelo roda ponto-a-ponto<br>recebendo apenas P · sem Q_obs<br>como entrada",
                          showarrow=False, font=dict(size=11))
    # Início e fim
    fig_cs.add_annotation(x=0, y=0.75, text="Início:<br>31/03/2025", showarrow=False, font=dict(size=10))
    fig_cs.add_annotation(x=480, y=0.75, text="Começa<br>a avaliar", showarrow=False, font=dict(size=10))
    fig_cs.add_annotation(x=6600, y=0.75, text="Fim:<br>30/12/2025", showarrow=False, font=dict(size=10))

    fig_cs.update_layout(
        template="plotly_white",
        xaxis=dict(title="Horas no período de teste", range=[-200, 6800]),
        yaxis=dict(range=[0.4, 2.3], showticklabels=False, showgrid=False),
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown(
        r"""
        **Como funciona:**
        - O modelo **não é reiniciado** a cada passo
        - Roda do início ao fim do período de teste **em cadeia**, alimentado
          apenas por **precipitação** (zero vazão observada)
        - **Warmup de 480 h** (~20 dias) no início: o estado interno da LSTM
          se estabiliza, mas essas horas não entram no cálculo das métricas
        - O NSE é calculado sobre a **série inteira** (5.526 pontos válidos
          nos 9 meses de teste)

        **Por que isso é importante:** expõe **acúmulo de erros**, **drift**
        e **capacidade de reproduzir o regime completo** (baixas, médias,
        altas). É o teste **mais severo**, típico de modelos usados para
        **gestão de recursos hídricos** e **curvas de permanência**.

        **Aplicação operacional:** análise de disponibilidade hídrica, outorga,
        dimensionamento de reservatórios, séries sintéticas para regionalização.
        """
    )

    st.divider()

    # --- Comparação em tabela
    st.markdown("##### Comparação lado a lado")

    st.markdown(
        """
        | Aspecto | Forecasting multi-horizonte | Simulação contínua |
        |---|---|---|
        | **Pergunta respondida** | "A que distância o modelo consegue prever?" | "O modelo representa o regime inteiro?" |
        | **Formato da saída** | Matriz (n_janelas × 24) — múltiplas previsões | Série temporal (n_pontos,) — uma trajetória |
        | **Input por predição** | 240 h de P a cada nova janela | P contínua do período de teste |
        | **Reset entre predições** | Sim (independentes) | **Não** |
        | **Horizonte avaliado** | 1h, 3h, 6h, 12h, 24h (separadamente) | Série toda |
        | **Melhor modelo (Fase 1)** | LSTM_TTD_Base → **NSE 0,84 @ 6h** | LSTM_TTD_Base_Fixed → **NSE 0,82** |
        | **Aplicação operacional** | Alerta, defesa civil, reservatórios | Disponibilidade, outorga, regionalização |

        **Por que os valores diferem?** Os dois modos testam coisas diferentes:
        forecasting é mais **fácil** (o modelo "vê" o passado recente a cada janela);
        contínua é mais **severa** (erros acumulam). A caracterização do
        **trade-off entre os dois modos** — parâmetros aprendíveis vencem em
        forecasting, fixos vencem em contínua — é um dos principais
        achados da Fase 1.
        """
    )

# -------------------------------------------------------- Estudo comparativo & Hipóteses
with tab_abl:
    st.subheader("Estudo de estudo comparativo — 10 configurações")

    st.markdown(
        """
        A pergunta central do estudo é: **"Quando a água chega?"**. A
        abordagem compara sistematicamente diferentes configurações dos
        módulos hidrológicos (TTD e SCS) com parâmetros fixos ou
        ajustáveis, permitindo isolar a contribuição de cada elemento.

        | # | Modelo | Espacial | TTD | SCS | Parâmetros |
        |---|---|---|---|---|---|
        | 1 | LSTM_Lumped | Concentrado | — | — | — |
        | 2 | LSTM | Distribuído | — | — | — |
        | 3 | LSTM_TTD_Base_Fixed | Distribuído | Maidment | — | Fixos |
        | 4 | LSTM_TTD_Base | Distribuído | Maidment | — | Ajustáveis |
        | 5 | LSTM_TTD_Manning_Fixed | Distribuído | Manning | — | Fixos |
        | 6 | LSTM_TTD_Manning | Distribuído | Manning | — | Ajustáveis |
        | 7 | LSTM_TTD_Base_SCS_Fixed | Distribuído | Maidment | Sim | Fixos |
        | 8 | LSTM_TTD_Base_SCS | Distribuído | Maidment | Sim | Ajustáveis |
        | 9 | LSTM_TTD_Manning_SCS_Fixed | Distribuído | Manning | Sim | Fixos |
        | 10 | LSTM_TTD_Manning_SCS | Distribuído | Manning | Sim | Ajustáveis |
        """
    )

    st.markdown(
        """
        #### Cinco hipóteses operacionais (H1–H5)

        **H1 — Distribuído > Concentrado**
        A configuração distribuída (precipitação, CN e $T_c$ por ottobacia)
        deve superar a concentrada (*lumped*), pois preserva a
        heterogeneidade espacial da bacia. *(Modelo 2 > Modelo 1)*

        **H2 — Manning > Base para $T_c$**
        O tempo de concentração ajustado pela rugosidade do uso do solo
        deve superar o $T_c$ puramente topográfico, pois incorpora
        informação adicional da superfície. *(Modelos 5–6 > Modelos 3–4)*

        **H3 — Parâmetros ajustáveis > fixos**
        Parâmetros físicos otimizados via *backpropagation* devem superar
        valores fixos da literatura, por permitirem adaptação às
        características específicas da bacia. *(Modelos 4, 6, 8, 10 >
        Modelos 3, 5, 7, 9)*

        **H4 — SCS-CN agrega valor**
        A inclusão do SCS-CN deve melhorar o desempenho quando combinada
        com o TTD, por representar a transformação chuva-escoamento de
        forma fisicamente consistente. *(Modelos 7–10 > Modelos 3–6)*

        **H5 — Modelo completo > *baselines***
        O modelo híbrido proposto deve superar tanto o LSTM puro quanto
        a configuração concentrada, combinando representação física com
        capacidade de aprendizado. *(Modelos 9–10 > Modelos 1–2)*
        """
    )

    st.info(
        "Os resultados dessas hipóteses são explorados em detalhe na "
        "página **Estudo comparativo & Trade-off**, incluindo o trade-off descoberto "
        "entre parâmetros aprendíveis (melhor previsão multi-horizonte) "
        "e fixos (melhor simulação contínua) — caracterização esta que "
        "não foi formulada *a priori* como hipótese."
    )

st.divider()

with st.expander("Referências-chave"):
    st.markdown(
        """
        - **Maidment, D. R. et al. (1996).** Unit hydrograph derived from a
          spatially distributed velocity field. *Hydrological Processes*,
          10(6), 831–844. — formulação do $T_c$ por campo de velocidades
        - **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term
          memory. *Neural Computation*, 9(8), 1735–1780. — LSTM
        - **Kratzert, F. et al. (2019).** Towards learning universal, regional,
          and local hydrological behaviors via machine learning applied to
          large-sample datasets. *HESS*, 23, 5089–5110. — LSTM em hidrologia
        - **Shen, C. et al. (2023).** Differentiable modeling to augment
          machine learning for scientific discovery. *Nature Reviews Earth
          & Environment*. — paradigma diferenciável
        - **Feng, D. et al. (2022/2024).** $\\delta$HBV: differentiable
          implementation of HBV.
        - **Woodward, D. E. et al. (2003).** Runoff curve number method:
          examination of the initial abstraction ratio.
        - **Vallejunior, L. F. et al. (2019).** Coeficiente de abstração
          inicial em bacias tropicais brasileiras. *RBRH*.
        - **Nearing, G. et al. (2024).** Global prediction of extreme floods
          in ungauged watersheds. *Nature*, 627, 559–563.
        - **ANA. (2025).** Nota Técnica nº 9/2025/COMUC/SHE — Produto
          BHAE_CN-2022.
        - **Laipelt, L. et al. (2024).** ANADEM v1 — Modelo Digital de
          Terreno para a América do Sul. IPH/UFRGS + ANA.
        """
    )
