import os
from ir_system import IR_System
from embedder import BertEmbedder, GloveEmbedder, TrainedWord2VecEmbedder, Word2VecEmbedder
from word2vec_trainer import load_json
import preprocessor
import glove_trainer
import subprocess

def setup_glove():
    print("Cloning glove repository...")
    subprocess.run(["git", "clone", "http://github.com/stanfordnlp/glove"])

    # copy demo.sh to glove/demo.sh
    with open("./demo.sh", 'r') as file:
        demo = file.read()

    with open("./glove/demo.sh", 'w') as file:
        file.write(demo)

    print("Creating my_corpus...")
    glove_trainer.main()

    print("Running glove make and demo.sh...")
    subprocess.run(["make"], cwd="./glove")
    subprocess.run(["./demo.sh"], cwd="./glove")


def setup_dataset():
    print("Preprocessing dataset...")
    preprocessor.main()
    print("Preprocessing Done.")



def main():  
    # check if the dataset is already preprocessed
    if not os.path.exists("./resources/cranfield_preprocessed.json"):
        setup_dataset()
    
    dataset = load_json("./resources/cranfield_preprocessed.json")
    docs = dataset["docs"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]


    # check if my_corpus exists. if not, train the glove model, and run the glove make and ./demo.sh
    # check if glove folder exists
    if not os.path.exists("./glove"):
        setup_glove()

    word2vec_embedder = Word2VecEmbedder()
    word2vec_system = IR_System(embedder=word2vec_embedder,
                                docs=docs,
                                queries=queries,
                                qrels=qrels)
    print("word2vec result:", word2vec_system.evaluate())

    
    trained_w2v_embedder = TrainedWord2VecEmbedder()
    trained_w2v_system = IR_System(embedder=trained_w2v_embedder,
                                docs=docs,
                                queries=queries,
                                qrels=qrels)
    print("trained word2vec result:", trained_w2v_system.evaluate())


    glove_embedder = GloveEmbedder()
    glove_system = IR_System(embedder=glove_embedder,
                            docs=docs,
                            queries=queries, 
                            qrels=qrels)
    print("glove result:", glove_system.evaluate())
    

    # bert model works significantly slower than the other models.
    # uncomment the following lines to run the bert model.
    """bert_embedder = BertEmbedder()
    bert_system = IR_System(embedder=bert_embedder, 
                            docs=docs, 
                            queries=queries, 
                            qrels=qrels)
    print("bert result:", bert_system.evaluate())"""


if __name__ == '__main__':
    main()



