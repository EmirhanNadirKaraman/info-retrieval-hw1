import math

import numpy as np
from numpy.linalg import norm

from embedder import Embedder


class IR_System:
    def __init__(
        self, embedder: Embedder, docs,
        queries, qrels
    ) -> None:
        self.embedder = embedder
        self.docs = docs
        self.queries = queries
        self.qrels = qrels

        self.doc_vecs, self.doc_map = self.get_doc_vecs()
        self.query_vecs, self.query_map = self.get_query_vecs()


    # Returns the average of DCG for each query in the dataset
    def evaluate(self) -> float:
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
    def get_scores(self, query: dict): 
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
        
        res = []
        for index, doc_text in enumerate(doc_texts):
            if index % 20 == 0:
                print(f"doc {index}")
            res.append(self.embedder.get_embedding(doc_text))

        return np.array(res), doc_map
    

    def get_query_vecs(self):
        query_map = dict()
        query_texts = [self.queries[query_id]["text"] for query_id in self.queries]

        counter = 0
        for query_id in self.queries:
            query_map[query_id] = counter
            counter += 1

        res = []
        for index, query_text in enumerate(query_texts):
            if index % 20 == 0:
                print(f"query {index}")
            res.append(self.embedder.get_embedding(query_text))

        return np.array(res), query_map


    # Returns cosine similarity between concatenation of document's text and title 
    # and filtered query text
    def score(self, doc: dict, query: dict) -> float:
        # This is to prevent calculation of the same thing multiple times. 
        # The values are retrieved from the precalculated dicts. 
        doc_vec = self.doc_vecs[self.doc_map[doc["doc_id"]]]
        query_vec = self.query_vecs[self.query_map[query["query_id"]]]

        """doc_text = doc["title"] + doc["text"]
        doc_vec = self.embedder.get_embedding(doc_text)
        query_vec = self.embedder.get_embedding(query["text"])"""

        cos_sim = np.dot(doc_vec, query_vec) / (norm(doc_vec)*norm(query_vec)) if np.any(doc_vec) else 0.0
        
        
        return cos_sim
