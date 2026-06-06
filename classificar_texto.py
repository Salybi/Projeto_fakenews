import argparse
from pathlib import Path

import joblib

from src.preprocessamento import limpar_texto


RAIZ = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ / "modelos" / "modelo_naive_bayes.joblib"


def classificar(texto: str) -> str:
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Execute primeiro: python treinar_modelo.py"
        )

    modelo = joblib.load(CAMINHO_MODELO)
    texto_limpo = limpar_texto(texto)
    classe = modelo.predict([texto_limpo])[0]

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba([texto_limpo])[0]
        classes = modelo.classes_
        confiancas = dict(zip(classes, probabilidades))
        confianca = confiancas.get(classe, 0.0)
        return f"{classe} (confianca aproximada: {confianca:.2%})"

    return classe


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica um texto como Fake ou Fato.")
    parser.add_argument("texto", nargs="+", help="Texto que sera classificado.")
    args = parser.parse_args()

    texto = " ".join(args.texto)
    print(classificar(texto))


if __name__ == "__main__":
    main()
