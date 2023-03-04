import json
import nltk
import re
import sys

import ir_datasets
from nltk.corpus import stopwords


# Preprocesses the dataset and saves it in .json format to location.
# Running this script once is enough to generate preprocessed dataset.
def main():
    dataset = ir_datasets.load("cranfield")
    nltk.download('stopwords')
    nltk.download('punkt')
    location = "./resources/cranfield_preprocessed.json"
    
    preprocessed = preprocess(dataset)
    json_str = json.dumps(preprocessed, ensure_ascii=False, indent=3)

    with open(location, "w") as docs_file:
        docs_file.write(json_str)


# Preprocesses title and text field in each document in the dataset and return
# the result as a dictionary.
def preprocess(dataset) -> dict:
    res = {"docs": []}

    for doc in dataset.docs_iter():
        new_doc = {}
        
        new_doc["doc_id"] = doc.doc_id
        new_doc["title"] = __filter(doc.title)
        new_doc["text"] = __filter(doc.text)
        new_doc["author"] = doc.author
        new_doc["bib"] = doc.bib

        res["docs"].append(new_doc)

    return res

# ! Requires these two to be executed previously: 
#   - nltk.download('stopwords')
#   - nltk.download('punkt')
def __filter(text: str) -> str:
    # remove links, numbers and special characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    
    # remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]

    return "".join(token + " " for token in tokens)


if __name__ == "__main__":
    main()
    """
    nltk.download("stopwords")
    nltk.download("punkt")
    s = "hi my name is enis, i am 22 years old, my blog site: https://enis.com"
    print(__filter(s)) # --> hi name enis years old blog site
    """
    sys.exit(0)
