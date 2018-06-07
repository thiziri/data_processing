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
from utils import word2vec_intersect_dataset, random_vector
import pyndri
from scipy import spatial

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        top_k_terms.py [--intxt=<text>|--inTopics=<path>] <embedding_model> [--binary] <outputfolder> [--stop] [--n=<val>] [--dataset=<val2>] [--index_dataset=val3] [--k|--p] [--q] [--qr]
	    
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
	        --q    Use this option to make a filter of the neighbors according to the query|text centroid neighbors .
	        --qr    Use this option to make a re-ranking of the neighbors according to the query|text centroid neighbors .
	        
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
	print("Cleaning word embeddings ...")
	word2vec_intersect_dataset(pyndri.Index(args["--index_dataset"]), args["<embedding_model>"], args["<outputfolder>"],bool(args["--binary"]),args["--dataset"],algo)
	#open the new word2vec
	model=Word2Vec.load_word2vec_format(join(args["<outputfolder>"],"word2vec_of_"+args["--dataset"]), binary=False)
	#model=Word2Vec.load_word2vec_format(args["<embedding_model>"], binary=False)
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

	encountred={}
	file=None
	#encountred_random={}
	print("Collecting neighbors ...")
	for w in toProcess:
			for word in toProcess[w]:
				if word not in encountred:
					neighbors=[]
					if word in model.vocab:
						neighbors=model.most_similar(word, topn=int(args["--n"])) # get most similar words
					else:
						#if word not in encountred_random:
						#randomVect=random_vector(model.layer1_size) 
						#encountred_random[word]=randomVect
						#else: randomVect=encountred_random[word]
						#neighbors=model.similar_by_vector(randomVect, topn=int(args["--n"]), restrict_vocab=None)
						neighbors=[(word,1)] #just has one neighbor : itself
					encountred[word]=neighbors
			#print("encountred terms:",encountred)

			if bool(args["--q"])or bool(args["--qr"]):
				print("Neighbors will be filtred ...")
				vectors=[model[v] for v in toProcess[w] if v in model.vocab]
				centroid=sum([v for v in vectors])/len(vectors)

				if bool(args["--q"])and not bool(args["--qr"]):
					file=open(join(args["<outputfolder>"],args["--dataset"])+"wv.top-"+args["--n"]+".neighbors_byCentroidIntersection.txt","w")
					print("With intersection with a query|text centroid neighbors")
					centroid_neighbors=model.similar_by_vector(centroid, topn=int(args["--n"]), restrict_vocab=None)
					#print("centroid neighborhood:",centroid_neighbors)
					for wd in encountred:
						w_neighbors=encountred[wd]
						intersection=[t for t in set([q[0] for q in centroid_neighbors]) & set([n[0] for n in w_neighbors])]
						if intersection!=[]:
							new=[(m,s) for m,s in encountred[wd] if m in intersection]
							encountred[wd]=new
							#print("intersection:",intersection)
						else:
							encountred[wd]=[(wd,1)]
					#print("res encountred:",encountred)

				elif not bool(args["--q"])and bool(args["--qr"]):
					file=open(join(args["<outputfolder>"],args["--dataset"])+"wv.top-"+args["--n"]+".neighbors_byCentroidRe-ranking.txt","w")
					print("With re-ranking of the neighbors using a query|text centroid")
					for wd in encountred:
						w_neighbors=encountred[wd]
						for nei in w_neighbors:
							try:
								nei=nei[0],nei[1]*(1.0 - spatial.distance.cosine(model[nei[0]], centroid))
							except:
								nei=nei[0],nei[1]*(1.0)
						neighbors=sorted(w_neighbors, key=lambda x:(-x[1],x[0]))
						encountred[wd]=neighbors

			else:
				print("No filter to a word neighbors")
				file=open(join(args["<outputfolder>"],args["--dataset"])+"wv.top-"+args["--n"]+".neighbors.txt","w")
	for wd in encountred:
		word_neighbors=wd
		for t in encountred[wd]:
			w2=t[0]
			#if (prog.match(w2)):
			#	w2=w2.replace('.','')
			w_txt=w2.lower() #stem(algo,' '.join(escape(w).split()).lower())
			word_neighbors+='\t'+w_txt+":"+str(t[1])
		file.write(word_neighbors+"\n")
	file.close()
	print("Finished.")
