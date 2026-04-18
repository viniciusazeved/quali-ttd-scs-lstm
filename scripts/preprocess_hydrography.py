"""
Pre-processamento da hidrografia ANA e ottobacias nivel 5.

- hidrografia_main.gpkg  -- rios com area contribuinte >= 1000 km2
- hidrografia_paraiba_sul.gpkg  -- bacia do Paraiba do Sul completa
- ottobacias_br_n5.gpkg  -- ottobacias nivel 5 nacionais simplificadas
"""
from __future__ import annotations

from pathlib import Path
import time

import geopandas as gpd

BASE = Path("C:/Users/vinic/OneDrive/Shapes")
HIDRO_SRC = BASE / "DispH_5kv27nov20_Snirh_shp" / "DispH_v27nov20_Snirh.shp"
OTTO_SRC = BASE / "geoft_bho_ach_otto_nivel_05.gpkg"

OUT = Path(__file__).resolve().parents[1] / "data" / "hidrografia"
OUT.mkdir(parents=True, exist_ok=True)


def preprocess_hydrography() -> None:
    t0 = time.time()
    print(f"Lendo {HIDRO_SRC}...")
    gdf = gpd.read_file(HIDRO_SRC)
    print(f"  {len(gdf):,} features, CRS={gdf.crs}")

    # Filtro 1: rios principais (area contribuinte >= 1000 km2)
    print("\nFiltro 1: rios principais (nuareacont >= 1000 km2)...")
    main_rivers = gdf[gdf["nuareacont"].fillna(0) >= 1000].copy()
    print(f"  {len(main_rivers):,} trechos")

    main_rivers["geometry"] = main_rivers["geometry"].simplify(0.01, preserve_topology=True)
    main_rivers = main_rivers.to_crs(4326)

    path_main = OUT / "hidrografia_main.gpkg"
    main_rivers.to_file(path_main, driver="GPKG")
    print(f"  Salvo em {path_main} ({path_main.stat().st_size / 1e6:.1f} MB)")

    # Filtro 2: bacia do Paraiba do Sul — cobacia iniciando com '7'
    # (regiao hidrografica Atlantico Sudeste)
    print("\nFiltro 2: bacia do Paraiba do Sul + afluentes principais...")
    ps = gdf[
        gdf["cobacia"].astype(str).str.startswith("7")
        & (gdf["nuareacont"].fillna(0) >= 100)
    ].copy()
    print(f"  {len(ps):,} trechos (>= 100 km2 de area contribuinte)")

    ps["geometry"] = ps["geometry"].simplify(0.002, preserve_topology=True)
    ps = ps.to_crs(4326)

    path_ps = OUT / "hidrografia_paraiba_sul.gpkg"
    ps.to_file(path_ps, driver="GPKG")
    print(f"  Salvo em {path_ps} ({path_ps.stat().st_size / 1e6:.1f} MB)")


def preprocess_ottobacias() -> None:
    """Simplifica as ottobacias nivel 5 do Brasil."""
    t0 = time.time()
    print(f"\nLendo {OTTO_SRC}...")
    gdf = gpd.read_file(OTTO_SRC)
    print(f"  {len(gdf):,} features, CRS={gdf.crs}")

    # Renomear colunas e simplificar
    gdf = gdf.rename(columns={
        "wts_cd_pfafstetterbasin": "cobacia",
        "wts_cd_pfafstetterbasincodelevel": "nivel",
        "wts_gm_area": "area_km2",
    })

    # Simplificar geometrias (nivel 5 = grandes, tolerance de 0.01 deg OK)
    print("  Simplificando geometrias (tolerance=0.01 deg ~ 1 km)...")
    gdf["geometry"] = gdf["geometry"].simplify(0.01, preserve_topology=True)

    # Projetar para WGS84
    print("  Reprojetando para WGS84...")
    gdf = gdf.to_crs(4326)

    # Manter só colunas essenciais
    gdf = gdf[["cobacia", "nivel", "area_km2", "geometry"]].copy()

    path_otto = OUT / "ottobacias_br_n5.gpkg"
    gdf.to_file(path_otto, driver="GPKG")
    print(f"  Salvo em {path_otto} ({path_otto.stat().st_size / 1e6:.1f} MB)")
    print(f"\nTempo total ottobacias: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    t_all = time.time()
    preprocess_hydrography()
    preprocess_ottobacias()
    print(f"\n=== Tempo total geral: {time.time() - t_all:.1f}s ===")
