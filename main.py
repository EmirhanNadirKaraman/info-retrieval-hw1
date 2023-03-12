from ir_system import IR_System
from embedder import BertEmbedder, GloveEmbedder, TrainedWord2VecEmbedder, Word2VecEmbedder
from word2vec_trainer import load_json

def main():  
    dataset = load_json("./resources/cranfield_preprocessed.json")
    docs = dataset["docs"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    word2vec_embedder = Word2VecEmbedder()
    word2vec_system = IR_System(embedder=word2vec_embedder,
                                docs=docs,
                                queries=queries,
                                qrels=qrels)

    

    trained_w2v_embedder = TrainedWord2VecEmbedder()
    trained_w2v_system = IR_System(embedder=trained_w2v_embedder,
                                docs=docs,
                                queries=queries,
                                qrels=qrels)


    glove_embedder = GloveEmbedder()
    glove_system = IR_System(embedder=glove_embedder,
                            docs=docs,
                            queries=queries, 
                            qrels=qrels)


    bert_embedder = BertEmbedder()
    bert_system = IR_System(embedder=bert_embedder, 
                            docs=docs, 
                            queries=queries, 
                            qrels=qrels)


    print("word2vec result:", word2vec_system.evaluate())
    print("trained word2vec result:", trained_w2v_system.evaluate())
    print("glove result:", glove_system.evaluate())
    print("bert result:", bert_system.evaluate())


if __name__ == '__main__':
    main()



