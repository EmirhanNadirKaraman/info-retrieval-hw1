import json
import sys

from gensim.models import Word2Vec


def main():
    dataset = load_json("./var/cranfield_preprocessed.json")
    sentences = build_sentences(dataset)
    
    model = Word2Vec(sentences=sentences)
    model.save("./var/word2vec.model")


def build_sentences(dataset) -> list[list[str]]:
    res = []

    for doc in dataset["docs"]:
        content = []
        content += doc["title"].split()
        content += doc["text"].split()
        res.append(content)

    return res


def load_json(path) -> dict:
    with open(path) as json_file:
        json_str = json_file.read()

    return json.loads(json_str)


if __name__ == "__main__":
    main()
    sys.exit(0)
