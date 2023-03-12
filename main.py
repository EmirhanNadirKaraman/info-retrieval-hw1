from ir_system import IR_System
from embedder import BertEmbedder, Word2VecEmbedder
from word2vec_trainer import load_json

def main():  
    dataset = load_json("./resources/cranfield_preprocessed.json")
    # dataset = load_json("/content/drive/MyDrive/resources/cranfield_preprocessed.json")
    docs = dataset["docs"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    bert_embedder = BertEmbedder()
    bert_system = IR_System(embedder=bert_embedder, 
                            docs=docs, 
                            queries=queries, 
                            qrels=qrels)
    
    """
    word2vec_embedder = Word2VecEmbedder()
    word2vec_system = IR_System(embedder=word2vec_embedder,
                                docs=docs,
                                queries=queries,
                                qrels=qrels)
    """
    
    print(bert_system.evaluate())



if __name__ == '__main__':
    main()



