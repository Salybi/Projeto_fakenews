from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.dados import carregar_dataset
from src.preprocessamento import limpar_texto


RAIZ = Path(__file__).resolve().parent
DATASET_PADRAO = RAIZ / "dataset_vacinas_final.csv"
PASTA_MODELOS = RAIZ / "modelos"
PASTA_RESULTADOS = RAIZ / "resultados"


def treinar(caminho_dataset: Path = DATASET_PADRAO) -> None:
    df = carregar_dataset(caminho_dataset)
    df["texto_limpo"] = df["texto"].apply(limpar_texto)
    df = df[df["texto_limpo"].str.len() > 0].reset_index(drop=True)

    distribuicao = df["classe"].value_counts().to_dict()
    if len(distribuicao) < 2:
        raise ValueError("A base precisa ter textos das classes Fake e Fato.")

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        df["texto_limpo"],
        df["classe"],
        test_size=0.25,
        random_state=42,
        stratify=df["classe"],
    )

    modelo = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            ("naive_bayes", MultinomialNB(alpha=1.0, fit_prior=False)),
        ]
    )

    modelo.fit(x_treino, y_treino)
    predicoes = modelo.predict(x_teste)

    acuracia = accuracy_score(y_teste, predicoes)
    relatorio = classification_report(y_teste, predicoes, digits=3, zero_division=0)
    matriz = confusion_matrix(y_teste, predicoes, labels=["Fake", "Fato"])

    PASTA_MODELOS.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    joblib.dump(modelo, PASTA_MODELOS / "modelo_naive_bayes.joblib")

    matriz_df = pd.DataFrame(
        matriz,
        index=["Real Fake", "Real Fato"],
        columns=["Previsto Fake", "Previsto Fato"],
    )
    matriz_df.to_csv(PASTA_RESULTADOS / "matriz_confusao.csv", encoding="utf-8")

    predicoes_df = pd.DataFrame(
        {
            "texto_original": df.loc[x_teste.index, "texto"].values,
            "texto_limpo": x_teste.values,
            "classe_real": y_teste.values,
            "classe_prevista": predicoes,
        }
    )
    predicoes_df.to_csv(PASTA_RESULTADOS / "predicoes_teste.csv", index=False, encoding="utf-8")

    vetor = modelo.named_steps["tfidf"]
    classificador = modelo.named_steps["naive_bayes"]
    termos = vetor.get_feature_names_out()
    termos_importantes = []
    for indice_classe, classe in enumerate(classificador.classes_):
        maiores_indices = classificador.feature_log_prob_[indice_classe].argsort()[-15:][::-1]
        for posicao, indice_termo in enumerate(maiores_indices, start=1):
            termos_importantes.append(
                {
                    "classe": classe,
                    "posicao": posicao,
                    "termo": termos[indice_termo],
                    "peso_log": classificador.feature_log_prob_[indice_classe][indice_termo],
                }
            )
    pd.DataFrame(termos_importantes).to_csv(
        PASTA_RESULTADOS / "termos_importantes.csv",
        index=False,
        encoding="utf-8",
    )

    metricas = [
        "PROJETO FAKE NEWS X FATO - NAIVE BAYES",
        "",
        f"Dataset usado: {caminho_dataset.name}",
        f"Total de textos validos: {len(df)}",
        f"Distribuicao das classes: {distribuicao}",
        f"Treino: {len(x_treino)} textos",
        f"Teste: {len(x_teste)} textos",
        "",
        "Tecnica utilizada:",
        "TF-IDF para transformar textos em vetores + Multinomial Naive Bayes para classificacao.",
        "O parametro fit_prior=False foi usado para reduzir o efeito do desbalanceamento da base.",
        "",
        f"Acuracia: {acuracia:.3f}",
        "",
        "Matriz de confusao:",
        matriz_df.to_string(),
        "",
        "Relatorio de classificacao:",
        relatorio,
        "",
        "Observacao:",
        "A base final foi balanceada e documentada com links de checagens e fontes oficiais.",
        "Mesmo assim, o modelo deve ser interpretado como demonstracao academica, nao como verificador real de fatos.",
    ]
    (PASTA_RESULTADOS / "metricas.txt").write_text("\n".join(metricas), encoding="utf-8")

    print("\n".join(metricas))
    print(f"\nModelo salvo em: {PASTA_MODELOS / 'modelo_naive_bayes.joblib'}")
    print(f"Resultados salvos em: {PASTA_RESULTADOS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia o classificador Fake/Fato.")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PADRAO),
        help="Caminho do arquivo .json ou .csv com as colunas texto e tipo.",
    )
    args = parser.parse_args()
    treinar(Path(args.dataset))


if __name__ == "__main__":
    main()
