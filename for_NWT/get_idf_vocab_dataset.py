# Construct a file that contains a computed idf for all the vocabulary on the indexed dataset
import logging, sys, pprint
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
from os.path import join
import os.path
from gensim.models import Word2Vec
import numpy
import docopt
import pyndri

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        get_idf_vocab_dataset.py  <outputfolder> [--dataset=<val2>] [--index_dataset=<val3>] [--b=<val4>] 
	    
	    Options:
	        --dataset=<val2>	Precise the collection name that corresonds to the topics you are processing .
	        --index_dataset=val3   Provides the index of your data set to filter the word embeddings vocabulary while computing the neighbors .
	        --b    The b value of computing the alpha parameter of NWT model while computing the neighbors [default : 2].
	        
	    """)

	index=pyndri.Index(args["--index_dataset"])
	token2id,_,id2df=index.get_dictionary()
	file=open(join(args["<outputfolder>"],args["--dataset"])+"wv.idf.txt","w")
	b=2#args["--b"

	for word in token2id:
		alpha = (index.maximum_document()-id2df[token2id[word]]+0.5)/(id2df[token2id[word]]+0.5)+float(b)
		file.write(word+"\t"+str(alpha)+"\n")
	file.close()
	print("Finished.")
