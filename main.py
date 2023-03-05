import sister
import numpy as np
import gensim.downloader as api

from preprocessor import __filter


def wordtovec_embedding(model, sentence):
    filtered_sentence = __filter(sentence)

    # word2vec embedding of each word, combined in a 2d list
    result = np.array([model[w] for w in filtered_sentence.split() if w in model])

    # the mean value for each index in vectors
    mean = np.mean(result, axis=0)

    return mean.reshape(1, -1)


def main(): 
    # word2vec model pretrained with Google News dataset
    wv_model = api.load('word2vec-google-news-300')

    # pretrained bert model
    bert_embedder = sister.BertEmbedding(lang='en')

    sentence = "London is the capital of Great Britain"
    w2v_result = wordtovec_embedding(model=wv_model, sentence=sentence)
    bert_result = bert_embedder.embed(sentences=[sentence])

    print("word2vec result \n", w2v_result.shape, w2v_result)
    print("bert result \n", bert_result.shape, bert_result)


if __name__ == '__main__':
    main()