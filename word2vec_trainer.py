import json
import sys

from gensim.models import Word2Vec


def main():
    # changed paths because google drive does not support relative paths

    dataset = load_json("./resources/cranfield_preprocessed.json")
    # dataset = load_json("/content/drive/MyDrive/resources/cranfield_preprocessed.json")
    sentences = build_sentences(dataset)
    
    model = Word2Vec(sentences=sentences)
    model.save("./resources/word2vec.model")
    # model.save("/content/drive/MyDrive/resources/word2vec.model")


def build_sentences(dataset):
    sentences = []

    for doc in dataset["docs"].values():
        sentence = doc["title"].split() + doc["text"].split()
        sentences.append(sentence)

    return sentences


def load_json(path) -> dict:
    with open(path) as json_file:
        json_str = json_file.read()

    return json.loads(json_str)


if __name__ == "__main__":
    main()
    sys.exit(0)
