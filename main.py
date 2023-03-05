import sister
import numpy as np
import gensim.downloader as api

from preprocessor import Preprocessor

class Embedder:
    def __init__(self, preprocessor) -> None:
        print("in embedder")
        self.model = None
        self.preprocessor = preprocessor

    def get_embedding(self, sentence) -> np.array(float):
        """override this method"""
        pass


class BertEmbedder(Embedder):
    def __init__(self, preprocessor):
        print("in bert embedder")
        super().__init__(preprocessor)
        self.model = sister.BertEmbedding(lang='en')

    def get_embedding(self, sentence):
        filtered_sentence = self.preprocessor.filter(sentence)
        return self.model.embed(sentences=[sentence])
    
    def get_multiple_embeddings(self, sentences: list[str]):
        return self.model.embed(sentences=sentences)


class Word2VecEmbedder(Embedder):
    def __init__(self, preprocessor):
        print("in w2v embedder")

        super().__init__(preprocessor)
        self.model = api.load('word2vec-google-news-300')
    
    
    def get_embedding(self, sentence):
        filtered_sentence = self.preprocessor.filter(sentence)

        # word2vec embedding of each word, combined in a 2d list
        result = np.array([self.model[w] for w in filtered_sentence.split() if w in self.model])

        # the mean value for each index in vectors
        mean = np.mean(result, axis=0)

        return mean.reshape(1, -1)
    


def main(): 
    preprocessor = Preprocessor()

    # word2vec model pretrained with Google News dataset
    w2v_embedder = Word2VecEmbedder(preprocessor=preprocessor)

    # pretrained bert model
    bert_embedder = BertEmbedder(preprocessor=preprocessor)
    
    sentence = "London is the capital of Great Britain"
    w2v_result = w2v_embedder.get_embedding(sentence=sentence)
    bert_result = bert_embedder.get_embedding(sentence=sentence)

    print("word2vec result \n", w2v_result.shape)
    print("bert result \n", bert_result.shape)
    


if __name__ == '__main__':
    main()