import sister
import numpy as np
import gensim.downloader as api

from gensim.models import Word2Vec


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
    
    def get_multiple_embeddings(self, sentences):
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
    
    def get_multiple_embeddings(self, sentences):
        return np.array([self.get_embedding(sentence) for sentence in sentences])
    

class TrainedWord2VecEmbedder(Embedder):
    def __init__(self):
        self.model = Word2Vec.load("./resources/word2vec.model").wv

    
    def get_embedding(self, sentence): 
        # word2vec embedding of each word, combined in a 2d list

        # if sentence is empty, return a vector of zeros
        if sentence.split() == []:
            return np.zeros(100)
        
        result = np.array([self.model[w] if w in self.model else np.zeros(100) for w in sentence.split()])

        # the mean value for each index in vectors
        mean = np.mean(result, axis=0)

        return mean.flatten()
    

    def get_multiple_embeddings(self, sentences):
        return np.array([self.get_embedding(sentence) for sentence in sentences])

