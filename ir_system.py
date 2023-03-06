import math

from numpy import dot
from numpy.linalg import norm

class Embedder:
    def __init__(self) -> None:
        pass

    def get_embedding(sentence) -> float:
        """override this method"""
        pass


class IR_System:
    def __init__(
        self, embedder: Embedder, docs: list[dict],
        queries: list[dict], qrels: list[dict]
    ) -> None:
        self.embedder = embedder
        self.docs = docs
        self.queries = queries
        self.qrels = qrels


    # Returns the average of DCG for each query in the dataset
    def evaluate(self) -> float:
        total = 0
        done = 0
        for query_id, qrel in self.qrels.items():
            query = self.queries[query_id]
            total += self.dcg(query)
            done += 1

        return total / done

    
    # Discounted Cumulative Average for a query
    def dcg(self, query) -> float:
        scores = self.get_scores(query)
        res = 0
        for i, (_, doc) in enumerate(scores):
            relavence = self.qrels[query["query_id"]][doc["doc_id"]]["relevance"]
            res += (2**relavence  - 1) /  math.log(x=i + 2, base=2)

        return res

    
    # Returns a list of (score, doc)
    # score: cosine similarity of the doc and query
    # doc: dict representation of a document
    def get_scores(self, query: dict) -> list[tuple[float, dict]]: 
        scores = []

        for doc in self.docs:
            score = self.score(doc, query)
            scores.append(tuple(score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


    # Returns cosine similarity between concatination of document's text and tile 
    # and filtered query text
    def score(self, doc: dict, query: dict) -> float:
        doc_text = doc["title"] + doc["text"]

        doc_vec = self.embedder.get_embedding(doc_text)
        query_vec = self.embedder.get_embedding(query["text"])
        
        cos_sim = dot(doc_vec, query_vec) / (norm(doc_vec)*norm(query_vec))
        return cos_sim
