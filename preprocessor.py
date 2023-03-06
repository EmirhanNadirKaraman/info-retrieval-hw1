from collections import defaultdict
import json
import nltk
import re
import sys

import ir_datasets
from nltk.corpus import stopwords


# Preprocesses the dataset and saves it in .json format to location.
# Running this script once is enough to generate preprocessed dataset.
def main() -> None:
    dataset = ir_datasets.load("cranfield")
    location = "./resources/cranfield_preprocessed.json"

    try: 
        with open(location, "r") as docs_file:
            print("Preprocessed dataset already exists.")
            return
    except:
        nltk.download('stopwords')
        nltk.download('punkt')
        
        preprocessed = preprocess(dataset)
        json_str = json.dumps(preprocessed, ensure_ascii=False, indent=3)

        with open(location, "w") as docs_file:
            docs_file.write(json_str)
    

# Preprocesses title and text field in each document in the dataset and return
# the result as a dictionary.
def preprocess(dataset) -> dict:
    res = {}

    res["docs"] = preproc_docs(dataset)
    res["queries"] = preproc_queries(dataset)
    res["qrels"] = compile_qrels(dataset)
    
    return res

def preproc_docs(dataset) -> list[dict]:
    res = {}

    for doc in dataset.docs_iter():
        new_doc = {}
        new_doc["doc_id"] = doc.doc_id
        new_doc["title"] = __filter(doc.title)
        new_doc["raw_title"] = doc.title
        new_doc["text"] = __filter(doc.text)
        new_doc["raw_text"] = doc.text
        new_doc["author"] = doc.author
        new_doc["bib"] = doc.bib

        res[doc.doc_id] = new_doc
    
    return res

def preproc_queries(dataset) -> list[dict]:
    res = {}

    for query in dataset.queries_iter():
        new_query = {}
        new_query["query_id"] = query.query_id
        new_query["text"] = __filter(query.text)
        new_query["raw_text"] = query.text

        res[query.query_id] = new_query
        
    return res


def compile_qrels(dataset) -> list[dict]:
    res = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        new_qrel = {}
        new_qrel["query_id"] = qrel.query_id
        new_qrel["doc_id"] = qrel.doc_id
        new_qrel["relevance"] = qrel.relevance
        new_qrel["iteration"] = qrel.iteration
        
        res[qrel.query_id][qrel.doc_id] = new_qrel

    return res


# ! Execute below two lines once before calling this function
# nltk.download('stopwords')
# nltk.download('punkt')
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