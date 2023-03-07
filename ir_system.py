import math

import numpy as np
from numpy.linalg import norm

from embedder import BertEmbedder, Embedder, Word2VecEmbedder
from word2vec_trainer import load_json


class IR_System:
    def __init__(
        self, embedder: Embedder, docs: list[dict],
        queries: list[dict], qrels: list[dict]
    ) -> None:
        self.embedder = embedder
        self.docs = docs
        self.queries = queries
        self.qrels = qrels

        self.doc_vecs, self.doc_map = self.get_doc_vecs()
        self.query_vecs, self.query_map = self.get_query_vecs()


    # Returns the average of DCG for each query in the dataset
    def evaluate(self) -> float:
        print("evaluating")
        total = 0
        done = 0

        for query_id, _ in self.qrels.items():
            query = self.queries.get(query_id, None)
            total += self.dcg(query)
            done += 1


        return total / done
    

    # Discounted Cumulative Average for a query
    def dcg(self, query) -> float:
        if not query: 
            return 0.0
        
        scores = self.get_scores(query)
        res = 0
        for i, (_, doc) in enumerate(scores):
            relevance = self.qrels[query["query_id"]].get(doc["doc_id"], {}).get("relevance", 0)
            res += (2**relevance  - 1) /  math.log(i + 2, 2)

        return res
    
    

    
    # Returns a list of (score, doc)
    # score: cosine similarity of the doc and query
    # doc: dict representation of a document
    def get_scores(self, query: dict) -> list[tuple[float, dict]]: 
        scores = []

        for doc_id in self.docs:
            doc = self.docs[doc_id]
            score = self.score(doc, query)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores
    

    def get_doc_vecs(self): 
        doc_map = dict()

        doc_texts = [self.docs[doc_id]["title"] + self.docs[doc_id]["text"] for doc_id in self.docs]

        counter = 0
        for doc_id in self.docs:
            doc_map[doc_id] = counter
            counter += 1

        return self.embedder.get_multiple_embeddings(doc_texts), doc_map
    

    def get_query_vecs(self):
        query_map = dict()
        query_texts = [self.queries[query_id]["text"] for query_id in self.queries]

        counter = 0
        for query_id in self.queries:
            query_map[query_id] = counter
            counter += 1

        return self.embedder.get_multiple_embeddings(query_texts), query_map


    # Returns cosine similarity between concatenation of document's text and title 
    # and filtered query text
    def score(self, doc: dict, query: dict) -> float:
        if int(doc["doc_id"]) % 1000 == 0: 
            print("in score, query_id: ", query['query_id'], doc['doc_id'])

        doc_vec = self.doc_vecs[self.doc_map[doc["doc_id"]]]
        query_vec = self.query_vecs[self.query_map[query["query_id"]]]

        """doc_text = doc["title"] + doc["text"]
        doc_vec = self.embedder.get_embedding(doc_text)
        query_vec = self.embedder.get_embedding(query["text"])"""

        cos_sim = np.dot(doc_vec, query_vec) / (norm(doc_vec)*norm(query_vec)) if np.any(doc_vec) else 0.0

        return cos_sim


dataset = load_json("./resources/cranfield_preprocessed.json")
docs = dataset["docs"]
queries = dataset["queries"]
qrels = dataset["qrels"]

word2vec_embedder = Word2VecEmbedder()
word2vec_system = IR_System(embedder=word2vec_embedder,
                            docs=docs,
                            queries=queries,
                            qrels=qrels)


print("word2vec result:", word2vec_system.evaluate())


"""bert_embedder = BertEmbedder()
bert_system = IR_System(embedder=bert_embedder, 
                        docs=docs, 
                        queries=queries, 
                        qrels=qrels)
print("bert result:", bert_system.evaluate())"""
