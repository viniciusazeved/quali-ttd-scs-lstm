"""
Catalogo de estacoes fluviometricas da ANA.

Convencionais (vazao diaria) — ANAF via hydrobr:
  Carvalho & Braga (2020) — https://doi.org/10.5281/zenodo.3755065

Telemetricas (vazao horaria) — HidroInventario/ANA:
  http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario
  (snapshot local em data/estacoes/ — regenerar com
   Apresentacao/shp/_export_shapes.py)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

_ANAF_FLOW_URL = (
    "https://raw.githubusercontent.com/wallissoncarvalho/hydrobr/"
    "master/hydrobr/resources/ANAF_flow_stations.csv"
)

_STATIONS_DIR = Path(__file__).parent / "data" / "estacoes"

# Estados estrangeiros que aparecem no ANAF (o catalogo inclui estacoes de
# paises vizinhos da bacia amazonica — filtradas para o app).
_FOREIGN = {
    "ARGENTINA", "BOLIVIA", "PARAGUAI", "PERU",
    "GUIANA FRANCESA", "SURINAME", "COLOMBIA",
    "GUIANA", "VENEZUELA", "URUGUAI",
}


@st.cache_data(ttl=3600, show_spinner="Carregando catalogo ANA de vazao...")
def load_flow_catalog(only_brazil: bool = True) -> pd.DataFrame:
    """
    Carrega o catalogo ANAF de estacoes fluviometricas.

    Colunas: Name, Code, Type, DrainageArea, SubBasin, City, State,
    Responsible, Latitude, Longitude, StartDate, EndDate, NYD, MD,
    N_YWOMD, YWMD.

    Parameters
    ----------
    only_brazil : bool
        Se True, exclui estacoes de paises vizinhos.
    """
    df = pd.read_csv(_ANAF_FLOW_URL)
    df["Code"] = df["Code"].apply(lambda x: f"{int(x):08}")

    if only_brazil:
        df = df[~df["State"].fillna("").str.upper().isin(_FOREIGN)].copy()

    # Normalizar datas (podem vir com formatos diferentes)
    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")

    # Classificacao Type — no catalogo ANAF todos os valores sao 1
    # (fluviometrica). Mantemos a coluna mas nao expomos como filtro.
    return df.reset_index(drop=True)


@st.cache_data
def compute_quality_score(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas quality_score (0-1) e quality_label.

    Score composto:
        50% NYD (anos de dados, saturando em 40 — series de vazao tendem a
             ser mais longas que as de chuva)
        30% completude (100 - MD%, saturando em 50%)
        20% recencia (anos desde EndDate, saturando em 20)

    Classes:
        Excelente (>=0.75)  |  Moderada (0.5-0.75)  |  Limitada (<0.5)
    """
    df = catalog.copy()

    nyd = df["NYD"].fillna(0).astype(float)
    s_nyd = (nyd / 40).clip(upper=1.0)

    md = df["MD"].fillna(100).astype(float)
    s_md = (1.0 - md / 50).clip(lower=0.0, upper=1.0)

    end = pd.to_datetime(df["EndDate"], errors="coerce")
    current_year = datetime.now().year
    years_since = current_year - end.dt.year.fillna(1950)
    s_rec = (1.0 - years_since / 20).clip(lower=0.0, upper=1.0)

    df["quality_score"] = 0.5 * s_nyd + 0.3 * s_md + 0.2 * s_rec

    df["quality_label"] = pd.cut(
        df["quality_score"],
        bins=[-np.inf, 0.5, 0.75, np.inf],
        labels=["Limitada", "Moderada", "Excelente"],
    )
    return df


def get_states(catalog: pd.DataFrame) -> list[str]:
    return sorted(catalog["State"].dropna().unique())


def get_responsibles(catalog: pd.DataFrame) -> list[str]:
    return sorted(catalog["Responsible"].dropna().unique())


def get_sub_basins(catalog: pd.DataFrame) -> list[int]:
    return sorted(catalog["SubBasin"].dropna().unique().astype(int))


# ---------------------------------------------------------------------------
# Shapes locais (convencionais + telemetricas) — gerados em
# Apresentacao/shp/_export_shapes.py e copiados para data/estacoes/
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Carregando estacoes convencionais...")
def load_convencionais(only_ana: bool = False) -> pd.DataFrame:
    """
    Carrega shapefile local de estacoes fluviometricas convencionais
    (vazao diaria, leitura manual em regua ou linigrafo).

    Fonte: catalogo ANAF (Carvalho & Braga, 2020) — snapshot 03/2020.

    Parameters
    ----------
    only_ana : bool
        Se True, retorna apenas estacoes com Responsible='ANA' (RHN pura).
    """
    fname = "estacoes_convencionais_ana.shp" if only_ana else "estacoes_convencionais_todas.shp"
    gdf = gpd.read_file(_STATIONS_DIR / fname)
    gdf["StartDate"] = pd.to_datetime(gdf["StartDate"], errors="coerce")
    gdf["EndDate"] = pd.to_datetime(gdf["EndDate"], errors="coerce")
    # renomear de volta p/ nomes longos do ANAF
    gdf = gdf.rename(columns={
        "DrainArea": "DrainageArea",
        "Responsib": "Responsible",
        "QScore": "quality_score",
        "QLabel": "quality_label",
    })
    return pd.DataFrame(gdf.drop(columns="geometry"))


@st.cache_data(ttl=3600, show_spinner="Carregando estacoes telemetricas...")
def load_telemetricas(only_ana: bool = False) -> pd.DataFrame:
    """
    Carrega shapefile local de estacoes fluviometricas telemetricas
    (vazao horaria/sub-horaria, transmissao automatica).

    Fonte: ANA/HidroInventario — consulta em 04/2026, filtro
    tpEst=1 (fluviometrica) + telemetrica=1.

    Parameters
    ----------
    only_ana : bool
        Se True, retorna apenas estacoes da RHN (Origem='RHN').
    """
    fname = "estacoes_telemetricas_ana.shp" if only_ana else "estacoes_telemetricas_todas.shp"
    gdf = gpd.read_file(_STATIONS_DIR / fname)
    gdf["StartDate"] = pd.to_datetime(gdf["StartDate"], errors="coerce")
    gdf["EndDate"] = pd.to_datetime(gdf["EndDate"], errors="coerce")
    gdf = gdf.rename(columns={
        "DrainArea": "DrainageArea",
        "Responsib": "Responsible",
    })
    return pd.DataFrame(gdf.drop(columns="geometry"))


def get_origens(df: pd.DataFrame) -> list[str]:
    return sorted(df["Origem"].dropna().unique().tolist())


def get_status(df: pd.DataFrame) -> list[str]:
    return sorted(df["Status"].dropna().unique().tolist())
