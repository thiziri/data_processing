# Construct a file that contains a top-k similar words for each word in the input text using word embeddings
import logging, sys, pprint
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
from os.path import join
import os.path
from gensim.models import Word2Vec
import numpy
import docopt
import utils
from utils import extractTopics
from utils import stem
from utils import escape
import re
import nltk
from nltk.corpus import stopwords
from utils import word2vec_intersect_dataset, random_vector, knn
import pyndri

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        top_k_terms.py [--intxt=<text>|--inTopics=<path>] <embedding_model> [--binary] <outputfolder> [--stop] [--n=<val>] [--dataset=<val2>] [--index_dataset=<val3>] [--b=<val4>] [--k|--p]
	    
	    Options:
	        --intxt=<text>    Provide path to text file or just text between cotes .
	        --inTopics=<path>    Provide path to TREC topics file .
	        --stop    Choose this option to perform stop words removel for the input text .
	        --n=<val>    Number of desired neighbors for each word [default : 200] .
	        --dataset=<val2>	Precise the collection name that corresonds to the topics you are processing .
	        --index_dataset=val3   Provides the index of your data set to filter the word embeddings vocabulary while computing the neighbors .
	        --k    Choose this option to use krovetz stemmer .
	        --p    Choose this option to use porter stemmer .
	        --binary    Choose this option if you provide a binary file of word embedding space .
	        --b    The b value of computing the alpha parameter of NWT model while computing the neighbors [default : 2].
	        
	    """)

	txt=""
	toProcess={}
	if bool (args["--intxt"]):
		txt="the input text"
		toProcess[0]=args["--intxt"]
	elif bool(args["--inTopics"]):
		txt=args["--inTopics"]
		toProcess=extractTopics(args["--inTopics"])
	#print(toProcess)
	print("We will collect the top {k} neighbors for each word in {txt}\n".format(k=int(args["--n"]),txt=txt))
	if bool (args["--stop"]):
		print("With stop words removel")
	else:
		print("With NO stop words removel")

	algo=""
	if bool(args["--p"]):
		algo="porter"
	else:
		algo="krovetz"


	print("Please wait while we are collecting the {k} neighbors of each word ...  \n".format(k=int(args["--n"])))

	#print("\nWord2vec loading ...")
	#model=Word2Vec.load_word2vec_format(args["<embedding_model>"], binary=bool(args["--binary"]))
	#print("\nOK")
	#filtering with the collection vocabulary
	index=pyndri.Index(args["--index_dataset"])
	token2id,_,id2df=index.get_dictionary()
	print("Cleaning word embeddings ...")
	#word2vec_intersect_dataset(index, args["<embedding_model>"], args["<outputfolder>"],bool(args["--binary"]),args["--dataset"],algo)
	#open the new word2vec
	#model=Word2Vec.load_word2vec_format(join(args["<outputfolder>"],"word2vec_of_"+args["--dataset"]), binary=False)
	model=Word2Vec.load_word2vec_format("/users/iris/tbelkace/Project/2ndYear/prepared_data/word_neighbors/with_googleWE/clean_embeddings/word2vec_of_rob04", binary=False)
	print("Word embeddings OK.")

	# Processing of the input text before neighbors finding
	prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")

	for t in toProcess: 
		terms=toProcess[t].split() #stem(algo,toProcess[t]).split()
		text=""
		for w in terms:
			if (prog.match(w)):
				w=w.replace('.','')
				text=text+" "+w
		text=' '.join(escape(text).split())
		text=" ".join(nltk.word_tokenize(text))
		d=[]
		text=" ".join([stem(algo,w) for w in text.split()])
		d+= text.split()
		toProcess[t]=d

	#print(toProcess)

	stopWordsList=set(stopwords.words('english'))
	if bool(args["--stop"]):
		for i in toProcess:
			listWords=[w for w in toProcess[i] if w not in stopWordsList]
			toProcess[i]=listWords

	#print(toProcess)	

	file=open(join(args["<outputfolder>"],args["--dataset"])+"wv.top-"+args["--n"]+".neighbors.txt","w")

	#print(toProcess)

	encountred=[]
	#encountred_random={}
	print("Collecting neighbors ...")

	for w in toProcess:
		for word in toProcess[w]:
			if word not in encountred:
				encountred.append(word)
				if word in model.vocab:
					#neighbors=model.most_similar(word, topn=int(args["--n"])) # get most similar words using the word2vec function
					alpha = (index.maximum_document()-id2df[token2id[word]]+0.5)/(id2df[token2id[word]]+0.5)+float(args["--b"])
					neighbors=knn(word,alpha,model,int(args["--n"])) # get most similar words using the knn function
				else:
					#if word not in encountred_random:
					#randomVect=random_vector(model.layer1_size) 
					#encountred_random[word]=randomVect
					#else: randomVect=encountred_random[word]
					#neighbors=w2v.similar_by_vector(randomVect, topn=int(args["--n"]), restrict_vocab=None)
					neighbors=[(word,1)] #just has one neighbor
				#print(neighbors)
				word_neighbors=word
				for t in neighbors:
					w=t[0]
					#if (prog.match(w)):
					#	w=w.replace('.','')
					w_txt=w.lower() #stem(algo,' '.join(escape(w).split()).lower())
					word_neighbors+='\t'+w_txt+":"+str(t[1])
				file.write(word_neighbors+"\n")
	file.close()
	print("Finished.")
