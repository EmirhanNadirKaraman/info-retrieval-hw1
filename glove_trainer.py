from word2vec_trainer import load_json

def main():
    dataset = load_json("./resources/cranfield_preprocessed.json")
    word_list = list_words(dataset)
    joined_str = ' '.join(word_list)
    
    with open("./glove/my_corpus", 'w') as file: 
        file.write(joined_str)


def list_words(dataset):
    word_list = []

    for doc in dataset["docs"].values():
        sentence = doc["title"].split() + doc["text"].split()
        word_list.extend(sentence)

    
    return word_list


if __name__ == '__main__':
    main()