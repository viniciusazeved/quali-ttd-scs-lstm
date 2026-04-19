"""
Streamlit de apresentacao — Qualificacao de doutorado TTD-SCS-LSTM.

Doutorando: Vinicius de Azevedo Silva
Orientador: Prof. Hugo de Oliveira Fagundes
Coorientador: Prof. Edevar Luvizotto Junior
Laboratorio: LAPLA (Laboratorio de Planejamento Ambiental)
Data da defesa: 22 de abril de 2026 — UNICAMP / FECFAU / DRH

Estrutura narrativa de apresentacao — segue a ordem dos capitulos da
qualificacao: motivacao (PUB) -> monitoramento -> trajetoria -> revisao ->
area de estudo -> dados -> metodologia -> resultados -> proximos passos.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="TTD-SCS-LSTM — Qualificação",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS — polish do layout + identidade visual
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-weight: 600; }
    h2 { font-weight: 500; margin-top: 1.5rem; }

    /* Cards de metric */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar — largura e espaçamento */
    section[data-testid="stSidebar"] {
        width: 290px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }

    /* Nav items: mais compactos e modernos */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        padding: 0.35rem 0.75rem;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        margin-bottom: 1px;
    }

    /* Cards decorativos na sidebar */
    .side-header {
        text-align: center;
        padding: 0.6rem 0 0.3rem 0;
    }
    .side-name {
        font-weight: 600;
        font-size: 1.02rem;
        color: #0f172a;
        margin: 0.35rem 0 0.1rem 0;
        line-height: 1.25;
    }
    .side-role {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    .side-block {
        font-size: 0.82rem;
        line-height: 1.55;
        color: #475569;
        padding: 0.25rem 0.25rem;
    }
    .side-block b { color: #1e293b; }
    .side-block .muted { color: #94a3b8; }

    .side-badge {
        text-align: center;
        padding: 0.55rem 0.5rem;
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        margin: 0.2rem 0;
    }
    .side-badge .label {
        font-size: 0.7rem;
        color: #2563eb;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .side-badge .date {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.92rem;
        margin-top: 0.1rem;
    }

    .side-sep {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Ordem narrativa seguindo a estrutura da qualificacao
pages = {
    "Abertura": [
        st.Page("app_pages/visao_geral.py", title="1. Capa & Síntese", icon="🏠"),
    ],
    "Contexto e Motivação": [
        st.Page("app_pages/introducao.py", title="2. Introdução — o problema PUB", icon="💡"),
        st.Page("app_pages/estacoes_ana.py", title="3. Monitoramento convencional", icon="🗺️"),
        st.Page("app_pages/estacoes_telemetricas.py", title="4. Monitoramento telemétrico", icon="📡"),
        st.Page("app_pages/trajetoria.py", title="5. Trajetória da pesquisa", icon="🧭"),
    ],
    "Fundamentação": [
        st.Page("app_pages/revisao.py", title="6. Revisão bibliográfica", icon="📚"),
    ],
    "Área de Estudo & Dados": [
        st.Page("app_pages/bacia.py", title="7. Bacia do Rio Preto", icon="🌿"),
        st.Page("app_pages/dados.py", title="8. Dados", icon="📊"),
    ],
    "Modelo Proposto": [
        st.Page("app_pages/metodologia.py", title="9. Metodologia", icon="⚙️"),
    ],
    "Resultados (Fase 1)": [
        st.Page("app_pages/resultados.py", title="10. Ablação & Trade-off", icon="📈"),
        st.Page("app_pages/explorador.py", title="11. Explorar hidrogramas", icon="🔍"),
        st.Page("app_pages/hiperparametros.py", title="12. Hiperparâmetros", icon="🧪"),
    ],
    "Próximos Passos": [
        st.Page("app_pages/fase2.py", title="13. Fase 2 & Cronograma", icon="🚀"),
        st.Page("app_pages/conclusoes.py", title="14. Conclusões", icon="🎯"),
    ],
}

with st.sidebar:
    # --- Cabeçalho com identidade visual (chuva → rio)
    st.markdown(
        """
        <div class="side-header">
            <div style="font-size: 1.8rem; line-height: 1; letter-spacing: 0.2rem;">
                🌧️ 🏞️
            </div>
            <div class="side-name">Vinícius de Azevedo Silva</div>
            <div class="side-role">Qualificação de Doutorado</div>
        </div>
        <hr class="side-sep">
        """,
        unsafe_allow_html=True,
    )

    # --- Bloco institucional
    st.markdown(
        """
        <div class="side-block">
            <b>UNICAMP</b> · FECFAU<br>
            DRH — Departamento de Recursos Hídricos<br>
            <span class="muted">LAPLA · Planejamento Ambiental</span>
        </div>
        <hr class="side-sep">
        """,
        unsafe_allow_html=True,
    )

    # --- Orientação
    st.markdown(
        """
        <div class="side-block">
            <b>Orientador</b><br>
            Hugo de Oliveira Fagundes<br>
            <span style="display:block; margin-top: 0.4rem;"></span>
            <b>Coorientador</b><br>
            Edevar Luvizotto Junior
        </div>
        <hr class="side-sep">
        """,
        unsafe_allow_html=True,
    )

    # --- Badge da data
    st.markdown(
        """
        <div class="side-badge">
            <div class="label">Defesa</div>
            <div class="date">22 de abril de 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0.2rem; color: #94a3b8; font-size: 0.75rem;
                    line-height: 1.45; text-align: center; margin-top: 0.5rem;">
            💡 Percorra as seções na ordem — segue a narrativa dos capítulos da qualificação.
        </div>
        """,
        unsafe_allow_html=True,
    )

nav.run()
