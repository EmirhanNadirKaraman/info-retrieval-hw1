import re

import nltk
from nltk.corpus import stopwords

class Embedder:
    def __init__(self) -> None:
        pass

    def get_embedding(sentence) -> list[float]:
        """override this method"""
        pass


class IR_System:
    def __init__(self, embedder, dataset, queries, qrels):
        nltk.download('stopwords')
        nltk.download('punkt')


    def evaluate() -> float:
        pass

    def __filter(text: str) -> str:
        # remove links, numbers and special characters
        text = re.sub(r"http\S+", "", text)
        text = re.sub("[^A-Za-z]+", " ", text)
        
        # remove stopwords
        tokens = nltk.word_tokenize(text)
        tokens = [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]

        return "".join(token + " " for token in tokens)