import sister
import numpy as np
import gensim.downloader as api


class Embedder:
    def __init__(self) -> None:
        self.model = None

    def get_embedding(self, sentence) -> np.array(float):
        """override this method"""
        pass

    def get_multiple_embeddings(self, sentences):
        """override this method"""
        pass


class BertEmbedder(Embedder):
    def __init__(self):
        self.model = sister.BertEmbedding(lang='en')

    def get_embedding(self, sentence):
        return self.model.embed(sentences=[sentence]).flatten()
    
    def get_multiple_embeddings(self, sentences: list[str]):
        return self.model.embed(sentences=sentences)


class Word2VecEmbedder(Embedder):
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')
    
    def get_embedding(self, sentence):
        # word2vec embedding of each word, combined in a 2d list

        # if sentence is empty, return a vector of zeros
        if sentence.split() == []:
            return np.zeros(300)
        
        result = np.array([self.model[w] for w in sentence.split() if w in self.model])

        # the mean value for each index in vectors
        mean = np.mean(result, axis=0)

        return mean.flatten()
    
    def get_multiple_embeddings(self, sentences: list[str]):
        return np.array([self.get_embedding(sentence) for sentence in sentences])



sentences = ["I don't know what my name or my purpose on this Earth is", 
             "But you sure as hell know what you're doing"]

"""embedder = Word2VecEmbedder()
word2vec = embedder.get_multiple_embeddings(sentences)

print(word2vec.shape)

bert_embedder = BertEmbedder()
bert = bert_embedder.get_multiple_embeddings(sentences)
print("shapes: ", bert.shape, word2vec.shape)"""

