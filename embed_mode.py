"""
Modo embed — quando a pagina e carregada com query param ?embed=1,
esconde sidebar, header e rodape do Streamlit para que o conteudo apareca
limpo dentro de um iframe (StoryMap ArcGIS).

Uso: chame maybe_hide_chrome() logo apos st.set_page_config() em cada page.

O modo nao afeta o app standalone — so se a URL tiver ?embed=1.
"""
from __future__ import annotations

import streamlit as st


def maybe_hide_chrome() -> bool:
    """
    Se a URL tiver ?embed=1, injeta CSS para esconder sidebar/header.
    Retorna True se o modo embed foi ativado.
    """
    params = st.query_params
    if params.get("embed") != "1":
        return False

    st.markdown(
        """
        <style>
          /* Esconder sidebar e controles de navegacao */
          section[data-testid="stSidebar"] { display: none !important; }
          div[data-testid="stSidebarNav"] { display: none !important; }
          button[data-testid="collapsedControl"] { display: none !important; }

          /* Esconder header, rodape e menu */
          header[data-testid="stHeader"] { display: none !important; }
          footer { display: none !important; }
          #MainMenu { display: none !important; }

          /* Remover decoracoes */
          [data-testid="stDecoration"] { display: none !important; }
          [data-testid="stStatusWidget"] { display: none !important; }

          /* Expandir conteudo para ocupar o iframe */
          div.block-container {
              padding-top: 0.5rem !important;
              padding-bottom: 0.5rem !important;
              padding-left: 1rem !important;
              padding-right: 1rem !important;
              max-width: 100% !important;
          }

          /* Compacta titulos para caber melhor em iframes estreitos */
          h1 { font-size: 1.6rem !important; line-height: 1.2 !important; }
          h2 { font-size: 1.3rem !important; margin-top: 0.6rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return True
