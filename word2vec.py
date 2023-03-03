import ir_datasets
import sister
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

LINE = '\n'

class ExampleClass: 
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.dataset = ir_datasets.load('cranfield')

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


sentence_embedding = sister.MeanEmbedding(lang="en")
sentence = "I am a dog."
vector = sentence_embedding(sentence)
print(vector)





