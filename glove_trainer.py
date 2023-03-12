"""from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)"""

from glove import Glove

#Creating a corpus object

#Training the corpus to generate the co-occurrence matrix which is used in GloVe
corpus.fit(lines, window=10)

glove = Glove(no_components=5, learning_rate=0.05) 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
