import re
import unicodedata


STOPWORDS_PORTUGUES = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em",
    "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta",
    "estao", "estas", "estava", "este", "estes", "eu", "foi", "foram",
    "ha", "isso", "isto", "ja", "lhe", "lhes", "mais", "mas", "me",
    "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nas",
    "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num",
    "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos",
    "por", "porque", "qual", "quando", "que", "quem", "sao", "se", "sem",
    "seu", "seus", "sua", "suas", "tambem", "te", "tem", "tendo", "ter",
    "teu", "teus", "tua", "tuas", "um", "uma", "umas", "uns", "voce",
    "voces",
}


def remover_acentos(texto: str) -> str:
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(char for char in texto_normalizado if not unicodedata.combining(char))


def limpar_texto(texto: str) -> str:
    """Normaliza o texto para reduzir ruido antes da vetorizacao."""
    if not isinstance(texto, str):
        return ""

    texto = remover_acentos(texto.lower())
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^a-zA-Z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    palavras = [
        palavra
        for palavra in texto.split()
        if palavra not in STOPWORDS_PORTUGUES and len(palavra) > 2
    ]
    return " ".join(palavras)
