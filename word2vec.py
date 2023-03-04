import ir_datasets
import sister
from sister.word_embedders import Word2VecEmbedding
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

import gensim.downloader as api
# wv = api.load('word2vec-google-news-300')

filename = 'archive/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


print("model: ", model)

for index, word in enumerate(model.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.index_to_key)} is {word}")

# how to get to the next suggestion in github copilot?
print(model.index_to_key[0])

"""
# get the keys of wv
print(wv.keys())

print(type(wv))
print(type(Word2VecEmbedding()))
"""

model1 = gensim.models.Word2Vec(data, min_count = 1,size = 100, window = 5, sg=0) 
print(model1)

LINE = '\n'

class ExampleClass: 
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.dataset = ir_datasets.load('cranfield')
        self.embedder = sister.MeanEmbedding(lang="en", word_embedder=model)

    def embed(self, sentence: str):
        return self.embedder.embed(sentence)

    def get_stopwords(self): 
        return self.stopwords
    
    def get_dataset(self):
        return self.dataset
    
    def get_document(self, doc_id: int):
        return self.dataset.docs_store().get(str(doc_id))
    
    def get_query(self, query_id: int):
        # get the query with id 5
        for query in self.dataset.queries_iter():
            if query.query_id == str(query_id):
                return query
    
    def get_qrel(self, query_id: int, doc_id: int):
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id == str(query_id) and qrel.doc_id == str(doc_id):
                return qrel
            
    def get_multiple_queries(self, query_count: int = -1):
        print(LINE)

        query_list = []

        ctr = 0
        for query in self.dataset.queries_iter():
            query_list.append(query)
            ctr += 1
            if ctr == query_count: 
                break
        
        return query_list
    
    
    # query relevance
    def get_multiple_qrels(self, qrel_count: int = -1):
        print(LINE)

        qrel_list = []

        ctr = 0
        for qrel in self.dataset.qrels_iter():
            qrel_list.append(qrel)
            ctr += 1
            if ctr == qrel_count: 
                break
            
        return qrel_list

    def print_query(self, query_id: int):
        print(LINE)
        query = self.get_query(query_id)
        print(f"Query ID: {query.query_id}")
        print(f"Query Text: {query.text}")
        print(LINE)

    def print_document(self, doc_id: int):
        print(LINE)
        doc = self.get_document(doc_id)
        print(f"Document ID: {doc.doc_id}")
        print(f"Document Text: {doc.text}")
        print(LINE)
    

obj = ExampleClass()
stopwords = obj.get_stopwords()
dataset = obj.get_dataset()

print(stopwords[:5])
print(dataset)
print(obj.get_document(15))


vector = obj.embed(sentence="I am a dog.")
print(vector)
print(len(vector))
print(vector.shape)





