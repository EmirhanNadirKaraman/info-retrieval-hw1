import ir_datasets
from ir_datasets.formats.trec import TrecQrel
dataset = ir_datasets.load('cranfield')

# docs
for doc in dataset.docs_iter():
    print(doc)
    break

print("\n")

# queries
for query in dataset.queries_iter():
    print(query)
    break

print("\n")

# qrels
for qrel in dataset.qrels_iter():
    print(qrel)
    break


class Embedder:
    def __init__(self, preprocessor) -> None:
        self.preprocessor = preprocessor