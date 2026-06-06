import json
from pathlib import Path

import pandas as pd


MAPA_ROTULOS = {
    "fake": "Fake",
    "fake news": "Fake",
    "falso": "Fake",
    "falsa": "Fake",
    "verdadeiro": "Fato",
    "verdadeira": "Fato",
    "fato": "Fato",
    "true": "Fato",
}


def carregar_dataset(caminho: str | Path) -> pd.DataFrame:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    if caminho.suffix.lower() == ".json":
        with caminho.open("r", encoding="utf-8") as arquivo:
            conteudo = json.load(arquivo)
        registros = conteudo.get("dados", conteudo)
        df = pd.DataFrame(registros)
    elif caminho.suffix.lower() == ".csv":
        df = pd.read_csv(caminho, encoding="utf-8")
    else:
        raise ValueError("Use um arquivo .csv ou .json")

    if "tipo" not in df.columns and "classe" in df.columns:
        df["tipo"] = df["classe"]

    colunas_obrigatorias = {"texto", "tipo"}
    faltantes = colunas_obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {', '.join(faltantes)}")

    df = df.copy()
    df["texto"] = df["texto"].fillna("").astype(str)
    df["classe"] = (
        df["tipo"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(MAPA_ROTULOS)
    )
    df = df[df["classe"].isin(["Fake", "Fato"])]
    df = df.drop_duplicates(subset=["texto"]).reset_index(drop=True)
    return df
