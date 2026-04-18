"""
Pagina de resultados — heatmap NSE, ranking, trade-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_loader as dl
from plots import (
    COLORS,
    PALETTE,
    plot_hydrograph,
    plot_nse_heatmap,
    plot_nse_ranking,
    plot_tradeoff,
)

st.title("Resultados — estudo comparativo dos 10 modelos")
st.caption(
    "Comparamos sistematicamente 10 versões do modelo (o que chamamos de "
    "*estudo comparativo* ou, tecnicamente, *ablação*) para isolar a "
    "contribuição de cada módulo — previsão multi-horizonte e simulação "
    "contínua."
)

with st.expander("📖 Antes de tudo: o que é *previsão multi-horizonte* vs *simulação contínua*?"):
    st.markdown(
        """
        O **mesmo modelo** é avaliado de duas formas — são **dois produtos
        operacionais** diferentes para a hidrologia:

        | | Previsão multi-horizonte | Simulação contínua |
        |---|---|---|
        | **O que faz** | Olha 240 h de chuva passada e prevê as próximas 24 h. A janela "desliza" no tempo. | Roda 9 meses em sequência, alimentado só por chuva, sem reiniciar. |
        | **Métrica** | NSE em cada passo adiante: 1h, 3h, **6h**, 12h, 24h | NSE sobre a série toda |
        | **Aplicação** | Alerta de cheias · defesa civil · reservatórios em tempo real | Disponibilidade hídrica · outorga · séries sintéticas |
        | **Melhor Fase 1** | LSTM_TTD_Base → **NSE 0,84 em 6 h** | LSTM_TTD_Base_Fixed → **NSE 0,82** |

        **Por que destacamos o horizonte 6h?** É onde o modelo tem o melhor
        desempenho. Em 1h–3h o modelo ainda depende demais do "efeito
        inércia" da vazão anterior; em 12h–24h a incerteza cresce (o NSE
        cai para 0,75 em 24h). **Seis horas é o ponto ótimo.**

        > Para ver diagramas visuais do funcionamento de cada modo, vá
        > em **Metodologia → aba Treinamento & Loss**.
        """
    )

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
try:
    ablation = dl.load_ablation_summary()
    ablation_full = dl.load_ablation_results()
    continuous = dl.load_continuous_summary()
    continuous_raw = dl.load_continuous_raw()
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao carregar resultados: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs principais
# ---------------------------------------------------------------------------
best_forecast = ablation.iloc[0]
best_continuous = continuous.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Melhor — previsão 6h", f"NSE = {best_forecast['NSE_6h']:.3f}",
          best_forecast["Modelo"].replace("LSTM_TTD_", ""))
k2.metric("Melhor — simulação contínua", f"NSE = {best_continuous['NSE']:.3f}",
          best_continuous["Modelo"].replace("LSTM_TTD_", ""))
k3.metric("Modelos Muito Bom (≥ 0,75)",
          f"{(ablation['NSE_6h'] >= 0.75).sum()} / {len(ablation)}")
k4.metric("Ganho vs baseline",
          f"{(ablation['NSE_6h'].max() - ablation['NSE_6h'].min()) * 100:.0f} pp",
          "NSE 6h")

st.divider()

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
tab_heat, tab_trade, tab_hip, tab_rank, tab_cont, tab_rec, tab_lim, tab_tab = st.tabs([
    "Heatmap NSE", "Trade-off", "Hipóteses H1–H5", "Ranking",
    "Simulação contínua", "Recomendações", "Limitações", "Tabelas"
])

with tab_heat:
    st.subheader("NSE por modelo × horizonte")
    st.plotly_chart(plot_nse_heatmap(ablation), use_container_width=True)
    st.caption(
        "A célula quanto mais verde, melhor o NSE. **Dois padrões distintos "
        "emergem:** (i) o **LSTM_TTD_Base** atinge o pico em **6 h** "
        "(NSE = 0,84) e decai em horizontes mais longos; (ii) o "
        "**LSTM_TTD_Manning_SCS** faz o oposto — **melhora progressivamente** "
        "com o horizonte até **NSE = 0,83 em 24 h**. Em geral, os modelos "
        "com TTD mantêm desempenho mais estável em horizontes longos, "
        "enquanto o LSTM puro degrada mais acentuadamente."
    )

with tab_trade:
    st.subheader("Trade-off — previsão 6h × simulação contínua")
    st.plotly_chart(plot_tradeoff(ablation, continuous), use_container_width=True)

    st.markdown(
        """
        **Interpretação:**

        - Pontos acima da diagonal pontilhada têm *melhor desempenho na
          simulação contínua* do que em previsão de 6h — comportamento
          estável, bom para aplicação operacional contínua.
        - Pontos abaixo da diagonal têm *melhor forecasting pontual* mas
          acumulam erros em simulação longa — potencial *overfitting*
          temporal dos parâmetros aprendíveis.
        - **LSTM_TTD_Base (aprendível)** lidera em previsão 6h mas cai
          na contínua; **LSTM_TTD_Base_Fixed** faz o inverso; e
          **LSTM_TTD_Manning** é robusto nos dois regimes.
        """
    )

with tab_hip:
    st.subheader("Avaliação das 5 hipóteses operacionais")
    st.caption(
        "As hipóteses H1–H5 foram formuladas no Capítulo 1 e testadas "
        "sistematicamente pelo estudo de estudo comparativo. Abaixo, o veredito "
        "com base nos resultados da Fase 1 (bacia única)."
    )

    abl_by_name = ablation.set_index("Modelo")

    def _nse(model: str, col: str = "NSE_6h") -> float:
        try:
            return float(abl_by_name.loc[model, col])
        except KeyError:
            return float("nan")

    # Veredito DUPLO — conforme Tab. resumo_hipoteses (Cap. 4 da quali)
    hypotheses = [
        {
            "id": "H1",
            "name": "Distribuído > Concentrado",
            "test": "LSTM vs LSTM_Lumped",
            "prev": ("✅", "Confirmada", "NSE 6h: 0,62 vs 0,54"),
            "cont": ("✅", "Confirmada", "NSE contínuo: 0,67 vs 0,61"),
            "summary": (
                "A configuração distribuída (245 ottobacias) supera a "
                "concentrada em **ambas as abordagens** — preserva a "
                "heterogeneidade espacial da bacia."
            ),
        },
        {
            "id": "H2",
            "name": "Manning > Base para $T_c$",
            "test": "LSTM_TTD_Manning vs LSTM_TTD_Base",
            "prev": ("❌", "Rejeitada", "NSE 6h: Manning 0,83 vs Base 0,84"),
            "cont": ("✅", "Confirmada", "NSE contínuo: Manning 0,81 vs Base 0,70"),
            "summary": (
                "Na previsão de 6 h, Base leva por margem mínima. "
                "Na **simulação contínua, Manning supera significativamente** "
                "(+15,7%), evidenciando que a rugosidade por uso do solo "
                "confere **robustez** para aplicações de longo prazo."
            ),
        },
        {
            "id": "H3",
            "name": "Parâmetros ajustáveis > fixos",
            "test": "Pares aprendível vs fixo",
            "prev": ("✅", "Confirmada", "média: +18,3% NSE 6h"),
            "cont": ("❌", "Rejeitada", "TTD Base: fixo 0,82 > ajust 0,70"),
            "summary": (
                "Parâmetros ajustáveis **otimizam a previsão**; parâmetros "
                "**fixos** atuam como regularização física e **generalizam "
                "melhor na contínua**. É o **trade-off central** da Fase 1 — "
                "não estava entre as hipóteses originais e motivou a discussão."
            ),
        },
        {
            "id": "H4",
            "name": "SCS-CN agrega valor",
            "test": "Com SCS vs sem SCS (em cada combinação)",
            "prev": ("❌", "Rejeitada", "SCS degrada 6–9,5% em 6 h"),
            "cont": ("⚠️", "Parcial", "só agrega em TTD_Base ajustável (+8,6%)"),
            "summary": (
                "Em bacia única, o SCS-CN é **parcialmente redundante**: a "
                "LSTM com 240 h de histórico já captura implicitamente a "
                "transformação chuva-escoamento. Mas o CN permanece na "
                "arquitetura pois será **crítico na Fase 2** — é o atributo "
                "que diferencia bacias com solos distintos."
            ),
        },
        {
            "id": "H5",
            "name": "Modelo completo > modelos de referência",
            "test": "TTD-SCS-LSTM vs LSTM puro / Lumped",
            "prev": ("✅", "Confirmada", "+35,5% vs LSTM; +55,6% vs Lumped"),
            "cont": ("✅", "Confirmada", "+22% em NSE contínuo"),
            "summary": (
                "O modelo híbrido supera os *baselines* em **ambas as "
                "abordagens** e em margem expressiva — demonstra que a "
                "física codificada agrega informação mesmo em bacia única. "
                "Expectativa: ganho ainda maior em aplicação multi-bacia (Fase 2)."
            ),
        },
    ]

    for h in hypotheses:
        st.markdown(f"##### **{h['id']}** — {h['name']}")
        c1, c2 = st.columns(2)
        with c1:
            icon, verdict, evidence = h["prev"]
            st.markdown(f"**Previsão (6h):** {icon} {verdict}")
            st.caption(evidence)
        with c2:
            icon, verdict, evidence = h["cont"]
            st.markdown(f"**Simulação contínua:** {icon} {verdict}")
            st.caption(evidence)
        st.markdown(h["summary"])
        st.markdown("")

    st.divider()
    st.info(
        "**Achado não previsto — trade-off forecast × contínuo.** A "
        "caracterização numérica deste trade-off (parâmetros aprendíveis "
        "melhoram previsão, parâmetros fixos melhoram simulação contínua) "
        "não estava entre as hipóteses originais. É o **resultado mais "
        "expressivo** da Fase 1 e motiva diretamente a Fase 2 "
        "(regionalização multi-bacia)."
    )


with tab_rank:
    h_choice = st.select_slider(
        "Horizonte",
        options=["NSE_1h", "NSE_3h", "NSE_6h", "NSE_12h", "NSE_24h"],
        value="NSE_6h",
        format_func=lambda x: x.replace("NSE_", ""),
    )
    st.plotly_chart(plot_nse_ranking(ablation, h_choice), use_container_width=True)

    st.caption(
        "Três padrões: (1) modelos *lumped* e LSTM puro formam o grupo "
        "inferior (NSE < 0,66); (2) modelos TTD sem SCS com parâmetros "
        "aprendíveis **dominam o horizonte de 6 h**; (3) o SCS-CN em geral "
        "**degrada** a previsão de curto prazo (6 a 9,5%), mas o caso "
        "específico **LSTM_TTD_Manning_SCS** se destaca em **24 h** "
        "(NSE = 0,83) — sugerindo que a combinação Manning + SCS com "
        "parâmetros aprendíveis modula melhor a abstração em horizontes longos."
    )

with tab_cont:
    st.subheader("Simulação contínua — hidrograma completo")
    st.caption(
        "O modelo recebe apenas precipitação no período de teste "
        "(2025-03-31 → 2025-12-30) e roda sem vazão observada. As métricas "
        "abaixo (NSE, R², RMSE, PBIAS) são calculadas sobre a **série "
        "completa**, ignorando apenas pontos com falha."
    )

    model_sel = st.selectbox(
        "Modelo para visualizar a simulação contínua",
        options=continuous["Modelo"].tolist(),
        index=0,
    )

    try:
        sim = dl.load_predictions(model_sel, mode="continuous")
        if sim and "observed" in sim and "predicted" in sim:
            obs = pd.Series(sim["observed"], index=pd.DatetimeIndex(sim["timestamps"]))
            pred = pd.Series(sim["predicted"], index=pd.DatetimeIndex(sim["timestamps"]))

            # Métricas da série completa
            mask = (~obs.isna()) & (~pred.isna())
            o = obs[mask].values
            p = pred[mask].values
            nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
            r2 = float(np.corrcoef(o, p)[0, 1]) ** 2
            rmse = float(np.sqrt(np.mean((o - p) ** 2)))
            pbias = float(100 * (p.sum() - o.sum()) / o.sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("NSE (série toda)", f"{nse:.3f}")
            m2.metric("R²", f"{r2:.3f}")
            m3.metric("RMSE", f"{rmse:.2f} m³/s")
            m4.metric("PBIAS", f"{pbias:+.2f}%")

            # Cobertura efetiva da simulação
            n_nan_sim = int(pred.isna().sum())
            n_total = len(pred)
            cov_pct = 100 * (1 - n_nan_sim / n_total)

            if n_nan_sim > 0:
                st.info(
                    f"ℹ️ **Cobertura da simulação:** {cov_pct:.1f}% "
                    f"({n_total - n_nan_sim:,}/{n_total:,} pontos). "
                    f"As lacunas visíveis no hidrograma abaixo vêm de **falhas no "
                    f"input de precipitação MERGE** (~54 pontos faltantes no período); "
                    f"como a LSTM usa lookback de 240 h, cada falha propaga por "
                    f"~10 dias à frente, gerando {n_nan_sim:,} pontos sem previsão "
                    f"(lacunas concentradas em abr–jul/2025). **Isso é limitação do "
                    f"dado, não do modelo.**"
                    .replace(",", ".")
                )

            st.plotly_chart(
                plot_hydrograph(obs, pred,
                                title=f"Simulação contínua — {model_sel}",
                                pred_label=f"Simulado — {model_sel}",
                                height=460),
                use_container_width=True,
            )
        else:
            st.warning(f"Dados de simulação contínua indisponíveis para {model_sel}.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao carregar simulação: {exc}")

    st.divider()

    # Resumo tabular — todos os 10 modelos
    st.markdown("**Resumo das métricas — série completa da simulação contínua**")
    st.dataframe(
        continuous.rename(columns={
            "NSE": "NSE", "KGE": "KGE", "RMSE": "RMSE (m³/s)",
            "PBIAS": "PBIAS (%)", "R2": "R²",
        }).style.format({
            "NSE": "{:.3f}", "KGE": "{:.3f}",
            "RMSE (m³/s)": "{:.2f}", "PBIAS (%)": "{:+.2f}", "R²": "{:.3f}",
            "n_samples": "{:,.0f}",
        }).background_gradient(subset=["NSE", "KGE"], cmap="RdYlGn", vmin=0.5, vmax=0.85),
        use_container_width=True, hide_index=True,
    )

with tab_rec:
    st.subheader("Recomendações por aplicação")
    st.markdown(
        """
        Conforme discutido no Capítulo 4 da qualificação, recomenda-se:
        """
    )

    rec1, rec2, rec3 = st.columns(3)

    with rec1:
        st.markdown(
            """
            #### 🌊 Para sistemas de **alerta de cheias**

            **LSTM_TTD_Base** com parâmetros ajustáveis

            **NSE₆ₕ = 0,84** · KGE = 0,83

            *Melhor desempenho em previsão de curto prazo, onde o
            retreinamento periódico permite aproveitar a flexibilidade
            dos parâmetros ajustáveis.*
            """
        )

    with rec2:
        st.markdown(
            """
            #### 💧 Para **simulação contínua**

            **LSTM_TTD_Base_Fixed** (ou LSTM_TTD_Manning)

            **NSE = 0,82** (Base Fixed) · 0,81 (Manning)

            *Parâmetros físicos fixos atuam como regularização,
            garantindo estabilidade ao longo de anos — ideal para
            disponibilidade hídrica, outorga e séries sintéticas.*
            """
        )

    with rec3:
        st.markdown(
            """
            #### 🎯 Para aplicação **robusta em ambos**

            **LSTM_TTD_Manning** (parâmetros ajustáveis)

            **NSE₆ₕ = 0,83** · **NSE contínuo = 0,81**

            *Degradação mínima (apenas 0,02 de NSE) entre as duas
            abordagens — a incorporação da rugosidade do uso do solo
            confere consistência aos dois regimes de avaliação.*
            """
        )

    st.divider()
    st.caption(
        "A escolha entre parâmetros fixos e ajustáveis **não é questão "
        "de superioridade técnica**, mas de adequação à aplicação "
        "operacional."
    )


with tab_lim:
    st.subheader("Limitações reconhecidas")
    st.caption(
        "Extraídas da Seção 4.8 da qualificação — contextualizam "
        "adequadamente os resultados e antecipam críticas da banca."
    )

    lim_data = [
        (
            "1. Validação em uma única bacia",
            "Os resultados se referem à bacia de Manuel Duarte (3.117 km², "
            "245 ottobacias). **A generalização será avaliada na Fase 2** "
            "com ~100 bacias brasileiras — validação leave-one-basin-out.",
        ),
        (
            "2. Período de dados de 5 anos",
            "O período 2021–2025 pode não capturar toda a variabilidade "
            "climática interanual. Além disso, o **teste** (abr–dez/2025) "
            "cobre predominantemente estação seca e transição para úmida, "
            "sem incluir o **pico da estação úmida (jan–mar)**. Na Fase 2, "
            "com bases mais longas, essa limitação será atenuada.",
        ),
        (
            "3. Escoamento de base implícito",
            "Os módulos TTD e SCS-CN representam **apenas o escoamento "
            "direto**. A componente lenta (*baseflow*) é capturada "
            "**implicitamente** pela memória de 240 h da LSTM. A "
            "separação explícita via filtro digital diferenciável "
            "(Eckhardt, 2005) é aprimoramento planejado.",
        ),
        (
            "4. IUH gaussiano (simétrico)",
            "O hidrograma unitário real é **assimétrico** (ascensão rápida, "
            "recessão lenta), mas usamos kernel gaussiano (simétrico) por "
            "parcimônia e estabilidade numérica. A LSTM compensa a "
            "assimetria residualmente. Substituição por distribuição gamma "
            "ou IUH discreto está no roadmap do Artigo 1.",
        ),
        (
            "5. Ausência de comparação com modelos externos",
            "A Fase 1 comparou apenas configurações internas do framework. "
            "Comparação com **GR4J, HBV, EA-LSTM e δHBV** está planejada "
            "para o Artigo 1 — essencial para contextualizar no estado "
            "da arte internacional.",
        ),
        (
            "6. Módulo hidrológico isolado não avaliado",
            "O desempenho de $Q_{physics}$ (saída do TTD antes da LSTM) "
            "**não foi avaliado isoladamente**. Isso impede **quantificar "
            "a contribuição corretiva da LSTM**. Análise dos resíduos "
            "($Q_{pred} - Q_{physics}$) está planejada para o Artigo 1.",
        ),
    ]

    for titulo, desc in lim_data:
        with st.expander(f"**{titulo}**"):
            st.markdown(desc)

    st.divider()
    st.info(
        "💡 **Por que essa página?** Porque a banca vai perguntar. "
        "Ter as limitações bem articuladas — e **as soluções já mapeadas** — "
        "é sinal de maturidade metodológica e argumento de defesa forte."
    )


with tab_tab:
    st.subheader("Estudo comparativo completa — tabela")
    st.dataframe(
        ablation.style.format({
            "NSE_1h": "{:.3f}", "NSE_3h": "{:.3f}", "NSE_6h": "{:.3f}",
            "NSE_12h": "{:.3f}", "NSE_24h": "{:.3f}",
            "Params": "{:,.0f}",
        }).background_gradient(
            subset=["NSE_1h", "NSE_3h", "NSE_6h", "NSE_12h", "NSE_24h"],
            cmap="RdYlGn", vmin=0.5, vmax=0.85,
        ),
        use_container_width=True, hide_index=True,
    )

    # Parâmetros aprendidos (só para os que têm)
    st.markdown("**Parâmetros aprendidos nos modelos *learnable***")
    rows = []
    for entry in ablation_full:
        lp = entry.get("learned_params", {}) or {}
        if not lp:
            continue
        rows.append({
            "Modelo": entry["model_name"],
            "tc_scale (α)": lp.get("tc_scale"),
            "σ (h)": lp.get("sigma"),
            "λ (SCS)": lp.get("lambda_scs"),
            "NSE 6h": entry["test_by_horizon"]["6h"]["nse"],
            "Épocas treinadas": entry.get("epochs_trained"),
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).style.format({
                "tc_scale (α)": "{:.3f}", "σ (h)": "{:.2f}",
                "λ (SCS)": "{:.3f}", "NSE 6h": "{:.3f}",
                "Épocas treinadas": "{:.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "**Interpretação física (conforme Cap. 4 da qualificação):** "
            "o $t_{c\\_scale} \\approx 1{,}2$–1,3 indica que a bacia **responde "
            "20–30% mais lentamente** que o estimado pelo método de Maidment, "
            "possível efeito de armazenamento temporário em planícies de "
            "inundação. Os valores maiores de $\\sigma$ (~4–5 h) refletem "
            "hidrogramas mais dispersos. O $\\lambda$ convergiu para 0,06–0,17, "
            "**consistente com ValleJunior et al. (2019)** para bacias tropicais "
            "brasileiras (mediana 0,045) — e substancialmente inferior ao "
            "padrão 0,20 de condições norte-americanas (Woodward et al., 2003)."
        )

st.divider()
st.caption(
    "Resultados obtidos em 22-23/jan/2026, com SEED = 42 e as configurações "
    "descritas na aba Metodologia. Arquivos brutos em `outputs/final/`."
)
