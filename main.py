import ir_datasets
dataset = ir_datasets.load('cranfield')

# print the dataset contents 
print(dataset)

count = 0
# print dataset docs
for doc in dataset.docs_iter():
    print(doc)
    count += 1
    if count >= 1: 
        break

print("\n")
        
# print dataset queries
count = 0
for query in dataset.queries_iter():
    print(query)
    count += 1
    if count > 5: 
        break

print("\n")

# print dataset qrels
count = 0
for qrel in dataset.qrels_iter():
    print(qrel)
    count += 1
    if count > 5: 
        break

print("qrels done")