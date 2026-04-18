"""
Carregamento de dados do projeto TTD-SCS-LSTM.

Centraliza acesso ao:
- Dataset HDF5 v2 (precipitacao, vazao, atributos das 245 ottobacias)
- Shapefiles: bacia Manuel Duarte + ottobacias CN 2022
- Resultados de ablacao, simulacao continua e busca de hiperparametros
- Predicoes (.npz) por modelo
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import streamlit as st

# Pasta local do Streamlit (dados pré-processados ou copiados para deploy)
STREAMLIT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA = STREAMLIT_ROOT / "data"

# Para deploy (Streamlit Cloud): se houver data/dataset_v2.h5 local, usa.
# Para dev: se nao houver, cai para os caminhos do projeto TTD-SCS-LSTM.
_has_local = (LOCAL_DATA / "dataset_v2.h5").exists()

if _has_local:
    DATA_HDF5 = LOCAL_DATA / "dataset_v2.h5"
    BACIA_SHP = LOCAL_DATA / "bacia" / "bacia_manuel_duarte.shp"
    OTTOBACIAS_GPKG = LOCAL_DATA / "ottobacias_cn_2022.gpkg"
    OUTPUTS_ABLATION = LOCAL_DATA / "outputs" / "final" / "ablation"
    OUTPUTS_CONTINUOUS = LOCAL_DATA / "outputs" / "final" / "continuous"
    OUTPUTS_HYPERPARAM = LOCAL_DATA / "outputs" / "hyperparam_search"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_HDF5 = PROJECT_ROOT / "data" / "processed" / "dataset_v2.h5"
    BACIA_SHP = PROJECT_ROOT / "data" / "raw" / "bacia" / "bacia_manuel_duarte.shp"
    OTTOBACIAS_GPKG = PROJECT_ROOT / "data" / "processed" / "ottobacias_cn_2022.gpkg"
    OUTPUTS_ABLATION = PROJECT_ROOT / "outputs" / "final" / "ablation"
    OUTPUTS_CONTINUOUS = PROJECT_ROOT / "outputs" / "final" / "continuous"
    OUTPUTS_HYPERPARAM = PROJECT_ROOT / "outputs" / "hyperparam_search"

# Hidrografia — sempre local (já pré-processada dentro do app)
HIDRO_DIR = LOCAL_DATA / "hidrografia"
HIDRO_MAIN = HIDRO_DIR / "hidrografia_main.gpkg"
HIDRO_PS = HIDRO_DIR / "hidrografia_paraiba_sul.gpkg"
OTTOS_BR = HIDRO_DIR / "ottobacias_br_n5.gpkg"


# ---------------------------------------------------------------------------
# Dataset HDF5
# ---------------------------------------------------------------------------
def _ts_to_datetime(arr: np.ndarray) -> pd.DatetimeIndex:
    """Converte array int64 (segundos Unix) para DatetimeIndex."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        # Heuristica: se valores sao muito grandes (>1e12), sao nanosegundos;
        # se estao na casa de 1e9, sao segundos Unix (epoch).
        if arr.size and arr.max() > 1e12:
            return pd.to_datetime(arr, unit="ns")
        return pd.to_datetime(arr, unit="s")
    return pd.to_datetime(arr)


@st.cache_data(show_spinner="Carregando dataset...")
def load_dataset_info() -> dict:
    """Sumario rapido do dataset_v2.h5."""
    with h5py.File(DATA_HDF5, "r") as f:
        info = {
            "n_ottobacias": int(f["ottobacia/area_km2"].shape[0]),
            "total_area_km2": float(np.array(f["ottobacia/area_km2"]).sum()),
            "splits": {},
        }
        for split in ("train", "val", "test"):
            if split in f:
                ts = _ts_to_datetime(np.array(f[f"{split}/timestamps"]))
                info["splits"][split] = {
                    "n_steps": len(ts),
                    "start": ts.min(),
                    "end": ts.max(),
                    "mean_q": float(np.array(f[f"{split}/streamflow"]).mean()),
                    "max_q": float(np.array(f[f"{split}/streamflow"]).max()),
                }
    return info


@st.cache_data(show_spinner="Carregando atributos das ottobacias...")
def load_ottobacia_attrs() -> pd.DataFrame:
    """Atributos fisicos das 245 ottobacias (area, CN, Tc)."""
    with h5py.File(DATA_HDF5, "r") as f:
        df = pd.DataFrame({
            "area_km2": np.array(f["ottobacia/area_km2"]),
            "cn_2022": np.array(f["ottobacia/cn_2022"]),
            "tc_base_h": np.array(f["ottobacia/tc_base_h"]),
            "tc_manning_h": np.array(f["ottobacia/tc_manning_h"]),
        })
    df.index.name = "ottobacia_id"
    return df


@st.cache_data(show_spinner="Carregando serie completa de vazao...")
def load_full_flow_series() -> pd.DataFrame:
    """Serie horaria completa: concatena train + val + test."""
    with h5py.File(DATA_HDF5, "r") as f:
        frames = []
        for split in ("train", "val", "test"):
            if split not in f:
                continue
            ts = _ts_to_datetime(np.array(f[f"{split}/timestamps"]))
            q = np.array(f[f"{split}/streamflow"])
            p = np.array(f[f"{split}/precipitation"]).mean(axis=1)
            frames.append(pd.DataFrame({
                "timestamp": ts,
                "streamflow": q,
                "precip_mean": p,
                "split": split,
            }))
    df = pd.concat(frames, ignore_index=True)
    df = df.set_index("timestamp").sort_index()
    return df


@st.cache_data(show_spinner="Carregando precipitacao distribuida...")
def load_precipitation_matrix(split: str = "test") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Retorna (timestamps, precip [T, 245])."""
    with h5py.File(DATA_HDF5, "r") as f:
        ts = _ts_to_datetime(np.array(f[f"{split}/timestamps"]))
        p = np.array(f[f"{split}/precipitation"])
    return ts, p


# ---------------------------------------------------------------------------
# Geoespacial
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando contorno da bacia...")
def load_bacia_geom() -> gpd.GeoDataFrame:
    """Retorna a geometria do contorno externo da bacia Manuel Duarte."""
    gdf = gpd.read_file(BACIA_SHP).to_crs(4326)
    return gdf


@st.cache_data(show_spinner="Carregando 245 ottobacias...")
def load_ottobacias_geom() -> gpd.GeoDataFrame:
    """GeoDataFrame das 245 ottobacias com CN 2022."""
    gdf = gpd.read_file(OTTOBACIAS_GPKG).to_crs(4326)
    return gdf


@st.cache_data(show_spinner="Carregando hidrografia nacional...")
def load_hydro_main() -> gpd.GeoDataFrame:
    """Rios principais do Brasil (area contribuinte >= 1000 km2)."""
    return gpd.read_file(HIDRO_MAIN)


@st.cache_data(show_spinner="Carregando hidrografia do Paraíba do Sul...")
def load_hydro_ps() -> gpd.GeoDataFrame:
    """Trechos da bacia do Paraiba do Sul (area contribuinte >= 100 km2)."""
    return gpd.read_file(HIDRO_PS)


@st.cache_data(show_spinner="Carregando ottobacias nacionais (nível 5)...")
def load_ottos_br() -> gpd.GeoDataFrame:
    """Ottobacias nacionais nivel 5 (16,5k features, simplificadas)."""
    return gpd.read_file(OTTOS_BR)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@st.cache_data
def load_ablation_summary() -> pd.DataFrame:
    """Sumario da ablacao (NSE por horizonte + numero de parametros)."""
    df = pd.read_csv(OUTPUTS_ABLATION / "summary.csv")
    df = df.sort_values("NSE_6h", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def load_ablation_results() -> list[dict]:
    """Lista com results.json de cada um dos 10 modelos da ablacao."""
    all_results = []
    for model_dir in sorted(OUTPUTS_ABLATION.iterdir()):
        if not model_dir.is_dir():
            continue
        results_path = model_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results.append(json.load(f))
    return all_results


@st.cache_data
def load_continuous_summary() -> pd.DataFrame:
    """Tabela resumo da simulacao continua (NSE, KGE, RMSE, PBIAS)."""
    with open(OUTPUTS_CONTINUOUS / "summary_continuous.json") as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        m = entry["metrics"]
        rows.append({
            "Modelo": entry["model_name"],
            "NSE": m["nse"],
            "KGE": m["kge"],
            "RMSE": m["rmse"],
            "PBIAS": m["pbias"],
            "R2": m["r2"],
            "n_samples": m["n_samples"],
        })
    df = pd.DataFrame(rows).sort_values("NSE", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def load_continuous_raw() -> list[dict]:
    """Dados completos da simulacao continua (inclui monthly_metrics)."""
    with open(OUTPUTS_CONTINUOUS / "summary_continuous.json") as f:
        return json.load(f)


@st.cache_data
def load_predictions(model_name: str, mode: str = "ablation") -> dict:
    """
    Carrega predictions.npz (ablation) ou simulation.npz (continuous).

    Ablation (predictions.npz): {pred: (N, 24), target: (N, 24)}
    Continuous (simulation.npz): {Q_sim: (T,), Q_obs: (T,), timestamps: (T,)}

    Retorna dict padronizado com chaves: observed, predicted, timestamps.
    """
    if mode == "ablation":
        path = OUTPUTS_ABLATION / model_name / "predictions.npz"
    else:
        path = OUTPUTS_CONTINUOUS / model_name / "simulation.npz"

    if not path.exists():
        return {}

    with np.load(path, allow_pickle=True) as data:
        raw = {k: data[k] for k in data.files}

    out: dict = {}
    if "pred" in raw:
        out["predicted"] = raw["pred"]
    elif "Q_sim" in raw:
        out["predicted"] = raw["Q_sim"]

    if "target" in raw:
        out["observed"] = raw["target"]
    elif "Q_obs" in raw:
        out["observed"] = raw["Q_obs"]

    if "timestamps" in raw:
        out["timestamps"] = _ts_to_datetime(raw["timestamps"])
    return out


@st.cache_data
def load_hyperparam_search() -> pd.DataFrame:
    """Resultados da busca de hiperparametros (108 configs)."""
    search_dirs = sorted(
        [d for d in OUTPUTS_HYPERPARAM.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not search_dirs:
        return pd.DataFrame()

    latest = search_dirs[0]
    csv_path = latest / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Display label → diretorio no disco
MODEL_LABELS: dict[str, str] = {
    "LSTM (lumped)": "LSTM_Lumped",
    "LSTM (distribuído)": "LSTM",
    "LSTM + TTD Base (fixo)": "LSTM_TTD_Base_Fixed",
    "LSTM + TTD Base (aprendível)": "LSTM_TTD_Base",
    "LSTM + TTD Manning (fixo)": "LSTM_TTD_Manning_Fixed",
    "LSTM + TTD Manning (aprendível)": "LSTM_TTD_Manning",
    "LSTM + TTD Base + SCS (fixo)": "LSTM_TTD_Base_SCS_Fixed",
    "LSTM + TTD Base + SCS (aprendível)": "LSTM_TTD_Base_SCS",
    "LSTM + TTD Manning + SCS (fixo)": "LSTM_TTD_Manning_SCS_Fixed",
    "LSTM + TTD Manning + SCS (aprendível)": "LSTM_TTD_Manning_SCS",
}


def get_model_dirs() -> list[str]:
    """Lista os nomes de diretorio dos modelos ordenada por desempenho."""
    return list(MODEL_LABELS.values())


def dir_to_label(model_dir: str) -> str:
    """Converte nome do diretorio para label amigavel."""
    for label, d in MODEL_LABELS.items():
        if d == model_dir:
            return label
    return model_dir
