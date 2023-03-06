import sister
import numpy as np
import gensim.downloader as api


class Embedder:
    def __init__(self) -> None:
        self.model = None

    def get_embedding(self, sentence) -> np.array(float):
        """override this method"""
        pass


class BertEmbedder(Embedder):
    def __init__(self):
        self.model = sister.BertEmbedding(lang='en')

    def get_embedding(self, sentence):
        return self.model.embed(sentences=[sentence])
    
    def get_multiple_embeddings(self, sentences: list[str]):
        return self.model.embed(sentences=sentences)


class Word2VecEmbedder(Embedder):
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')
    
    def get_embedding(self, sentence):
        # word2vec embedding of each word, combined in a 2d list
        result = np.array([self.model[w] for w in sentence.split() if w in self.model])

        # the mean value for each index in vectors
        mean = np.mean(result, axis=0)

        return mean.reshape(1, -1)
