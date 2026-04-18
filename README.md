# Qualificação de Doutorado — TTD-SCS-LSTM

Aplicação Streamlit interativa para apoiar a defesa da qualificação de
doutorado de **Vinícius de Azevedo Silva** (UNICAMP / FECFAU / DRH · LAPLA).

**Orientador:** Prof. Hugo de Oliveira Fagundes
**Coorientador:** Prof. Edevar Luvizotto Junior
**Data da defesa:** 22 de abril de 2026

## Conteúdo

Apresentação interativa em **13 seções narrativas**, cobrindo desde o
problema PUB (*Prediction in Ungauged Basins*) até as conclusões e
próximos passos da pesquisa, incluindo:

- Mapa interativo das ~2.000 estações fluviométricas da ANA
- Delineamento da sub-bacia do Rio Preto (Manuel Duarte, 3.117 km²)
  em 245 ottobacias
- Metodologia do modelo diferenciável TTD-SCS-LSTM
- Resultados do estudo comparativo dos 10 modelos — NSE 0,84 (previsão
  6 h) e NSE 0,82 (simulação contínua)
- Explorador interativo de hidrogramas observado × simulado
- Cronograma e próximos passos (Fase 2: regionalização multi-bacia)

## Como rodar

```powershell
# Instala dependências (primeira vez)
uv sync

# Roda o app
uv run streamlit run app.py
```

Abre em `http://localhost:8501`.

## Estrutura do projeto

```
streamlit_apresentacao/
├── app.py                    # Entry point + navegação
├── data_loader.py            # HDF5 + shapefiles + results.json
├── plots.py                  # Funções Plotly reusáveis
├── ana_client.py             # Cliente HidroWeb (vazão ANA)
├── stations.py               # Catálogo ANAF fluviométrico
├── app_pages/                # 13 páginas da apresentação
├── data/                     # Dados autossuficientes (gerado via script)
│   ├── dataset_v2.h5         # precipitação + vazão + atributos
│   ├── bacia/                # shapefile da bacia
│   ├── ottobacias_cn_2022.gpkg
│   ├── hidrografia/          # rios ANA + ottobacias Brasil
│   └── outputs/              # resultados treinados
├── scripts/
│   ├── prepare_deploy.py     # copia dados do projeto principal
│   └── preprocess_hydrography.py
├── pyproject.toml
└── .python-version
```

## Reproducibility

- Modelos treinados em 22–23/janeiro/2026 com `SEED = 42`
- Hardware: NVIDIA RTX 3000 Ada (8 GB)
- 300 épocas · early stopping patience 30 · batch 1.024

## Fontes de dados

- **Vazão:** ANA / HidroWeb — estação 58585000 (Manuel Duarte)
- **Precipitação:** MERGE (CPTEC/INPE) — 0,1° horário
- **DEM:** ANADEM v1 (IPH/UFRGS + ANA) — 30 m
- **CN:** BHAE_CN-2022 (ANA) — Nota Técnica nº 9/2025/COMUC/SHE
- **LULC:** MapBiomas Coleção 8.0 (ano-base 2022)
- **Hidrografia:** BHO 2017 v_01_05 (ANA)
- **Catálogo de estações:** ANAF via [hydrobr](https://github.com/wallissoncarvalho/hydrobr)
  (Carvalho & Braga, 2020 — [doi:10.5281/zenodo.3755065](https://doi.org/10.5281/zenodo.3755065))

## Licença

Código disponibilizado para fins acadêmicos.

---

*Desenvolvido com [Streamlit](https://streamlit.io/) e muito café.*
