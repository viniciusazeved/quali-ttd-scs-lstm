"""
Prepara o app para deploy no Streamlit Cloud.

Copia para `data/` dentro do app tudo que ele precisa para rodar
autossuficientemente (sem depender do projeto principal `D:/TTD_SCS_LSTM/`).

Uso:
    uv run python scripts/prepare_deploy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent.parent  # D:/TTD_SCS_LSTM

SRC_DATA = PROJECT_ROOT / "data"
SRC_OUTPUTS = PROJECT_ROOT / "outputs"
DST = APP_ROOT / "data"
DST.mkdir(parents=True, exist_ok=True)


def copy(src: Path, dst: Path) -> None:
    """Copia arquivo preservando diretórios."""
    if not src.exists():
        print(f"  [skip] nao encontrado: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [ok]   {src.name}  ({src.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    print(f"Projeto principal: {PROJECT_ROOT}")
    print(f"App:               {APP_ROOT}")
    print(f"Destino dos dados: {DST}\n")

    print("-- Dataset HDF5 + GPKG --")
    copy(SRC_DATA / "processed" / "dataset_v2.h5", DST / "dataset_v2.h5")
    copy(SRC_DATA / "processed" / "ottobacias_cn_2022.gpkg", DST / "ottobacias_cn_2022.gpkg")

    print("\n-- Shapefile da bacia --")
    for ext in ("cpg", "dbf", "prj", "sbn", "sbx", "shp", "shp.xml", "shx"):
        copy(
            SRC_DATA / "raw" / "bacia" / f"bacia_manuel_duarte.{ext}",
            DST / "bacia" / f"bacia_manuel_duarte.{ext}",
        )

    print("\n-- Outputs: estudo comparativo (ablation) --")
    src_abl = SRC_OUTPUTS / "final" / "ablation"
    dst_abl = DST / "outputs" / "final" / "ablation"
    for f in ("summary.csv", "all_results.json"):
        copy(src_abl / f, dst_abl / f)
    for model_dir in sorted(src_abl.iterdir()):
        if model_dir.is_dir():
            for f in ("predictions.npz", "results.json"):
                copy(model_dir / f, dst_abl / model_dir.name / f)

    print("\n-- Outputs: simulacao continua --")
    src_cont = SRC_OUTPUTS / "final" / "continuous"
    dst_cont = DST / "outputs" / "final" / "continuous"
    copy(src_cont / "summary_continuous.json", dst_cont / "summary_continuous.json")
    for model_dir in sorted(src_cont.iterdir()):
        if model_dir.is_dir():
            for f in ("simulation.npz", "results.json"):
                copy(model_dir / f, dst_cont / model_dir.name / f)

    print("\n-- Hyperparam search --")
    for search_dir in sorted((SRC_OUTPUTS / "hyperparam_search").iterdir()):
        if search_dir.is_dir():
            dst_sd = DST / "outputs" / "hyperparam_search" / search_dir.name
            for f in ("results.csv", "config.json", "results_partial.json"):
                copy(search_dir / f, dst_sd / f)

    # Tamanho total
    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"\n=== Total em data/: {total / 1e6:.1f} MB ===")


if __name__ == "__main__":
    main()
