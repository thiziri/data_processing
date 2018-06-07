# contains a set of used functions

import collections
from collections import defaultdict
import operator
from os import listdir
from os.path import join
import os.path
from krovetzstemmer import Stemmer
from porter2stemmer import Porter2Stemmer as stm
from gensim.models import Word2Vec
import numpy
from scipy import spatial
import re

""" 
Extract TREC topics on the pathTop parameter as dictionnary. 
ex: {0:"this is a text of the topic"}
"""

def extractTopics(pathTop):
	print("Extraction de : %s" %(pathTop))
	nb=0
	topics={}

	for f in listdir(pathTop):
		f = open(join(pathTop,f), 'r')   # Reading file
		l = f.readline().lower()
		# extracting topics
		while (l!=""):
			if (l!=""):
				num=0
				while((l.startswith("<num>")==False)and(l!="")) :
					l = f.readline().lower()
				num=l.replace("<num>","").replace("number:","").replace("\n","").replace(" ","")
				while ((l.startswith("<title>")==False)and(l!="")) :
					l = f.readline().lower()
				titre=""
				while((not l.startswith("</top>"))and(not l.startswith("<desc>"))and(l!="")):
					titre=titre+" "+l.replace("<title>","")
					l=f.readline().lower()
				if (titre!="" and num!=0):
					topics[num]=titre.replace("\n","").replace("topic:","").replace("\t"," ")
					nb+=1
			else : print("Fin.\n ")
		f.close()
		del f
		del l 
	return (collections.OrderedDict(sorted(topics.items())))


""" 
Translates each of the following characters to a blanc character in the input text.
ex: escape("U.S.") -> "U S "
"""

def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('#'): ' ',
        ord('_'): ' ',
        ord('@'): ' ',
        ord('='): None,
    })

""" 
Performs a stemming operation to the input text using the input algorithm in algo parameter.
ex: stem("krovetz","embeddings") -> "embed"
""" 
def stem(algo,text):
	if algo=="krovetz":
		stemmer = Stemmer()
		return(stemmer.stem(text))
	elif algo=="porter":
		s=stm()
		return(s.stem(text))
	else:
		print("ERROR STEMMING: {t} unkown.".format(t=algo))
		exit()


"""
Read different indexed terms in index with corresponding df (document frequency) and cf (collection frequency) of each term.
ex: a line in the resulting outFile corresponds to: "fort	1830	4935"  
"""
def indexToDfCf_file(index,outFile):
	print("\nIndex reading ... \n")

	token2id,_,id2df = index.get_dictionary()
	id2tf = index.get_term_frequencies()

	for w in token2id:
		outFile.write("{term}\t{df}\t{cf}\n".format(term=w,df=id2df[token2id[w]],cf=id2tf[token2id[w]]))
	outFile.close()

	print("DF CF reading finished.")

"""
Reads a document content of documents in the run file using the index.
ex: a resulting file will contain lines : "FT943-1053	234	extent:1 very:1 kleinwort:1 year:2	..."
"""
def indexToDocSet_file(index,outFile,run):
	print("Documents reading from run : "+run)
	in_run=open(run,"r")
	lines=in_run.readlines()
	document_run=[]
	for l in lines:
		if l!="\n":
			if l.split()[2] not in document_run:
				document_run.append(l.split()[2])
	_,id2token,_=index.get_dictionary()
	print(document_run)

	documents=[document_id for document_id in range(index.document_base(), index.maximum_document())]
	for id_d in documents:
		ext_Document, terms= index.document(id_d)
		if ext_Document in document_run:
			unique_terms=defaultdict(int)
			for w in terms:
				if w!=0:
					unique_terms[id2token[w]]+=1
			doc=ext_Document+"\t"+str(sum(unique_terms.values()))# external_document_id	length
			for w in unique_terms:
				doc=doc+"\t"+w+":"+str(unique_terms[w])
			outFile.write(doc+"\n")
	outFile.close()
	print("Run reading finished.")


"""
Cleans and stem a given word
"""
def cleanWord(word,stemming):
	prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
	if (prog.match(word)):
		word=word.replace('.','')
	word = stem(stemming,escape(word))

"""
Applies a filter to the word2vec model, such that vocabulary words that are not in the index_data are ignored.
Result, is an output file that contains the word embeddings of only words in your dataset.
(regardless the WE's size vectors)
"""
def word2vec_intersect_dataset(index_data, word2vec, output,ifBinary,dataSet,algo):
	w2v=Word2Vec.load_word2vec_format(word2vec, binary=ifBinary)
	token2id,_,_ = index_data.get_dictionary()
	out=open(join(output,"word2vec_of_"+dataSet),"w")
	words_to_keep={}
	word_count=0
	print("WE and Index loaded correctly")
	for w in w2v.vocab:
		w2=stem(algo,w)
		if w2 in token2id.keys():
			if w2 not in words_to_keep:
				words_to_keep[w2]=w2v[w]
				word_count+=1
			else:
				words_to_keep[w2]+=w2v[w]
	out.write(str(word_count)+" "+str(w2v.layer1_size)+"\n") # w2v.layer1_size gets the size of the embeddings
	for w in words_to_keep:
		out.write(w+" "+' '.join(str(e) for e in words_to_keep[w])+"\n")
	out.close()

"""
Initialize a vector of dim dimensions to real values in [-1,1]
"""
def random_vector(dim):
	x = numpy.zeros(dim)
	positions = numpy.random.choice(numpy.arange(dim), dim, replace=False)
	x[positions] = numpy.random.normal(0,1,2)
	return x
"""
It complete the absent words on the neigWordfile, comparing to the topics_file
"""
def complete_missing_words(topics_file,neigWord_file):
	words_inTopis=[]
	neiWord_file=[]
	with open(topics_file) as w:
		w_lines=w.readlines()
	w_lines=[x.strip() for x in w_lines]
	for l in w_lines:
		words=l.split()[1:]
		for wd in words:
			if wd not in words_inTopis:
				words_inTopis.append(wd)
	with open(neigWord_file) as w2:
		w_lines=w2.readlines()
	w_lines=[x.strip() for x in w_lines]
	words=[l.split()[0] for l in w_lines]

	missing=[w for w in words_inTopis if w not in words]
	#print(neiWord_file)
	if (len(missing))>0:
		print("There are some missing words of the queries. Program will exit.")
		exit()
	else:
		data=[]
		with open(neigWord_file) as in_file:
			for line in in_file:
				data += [line.split()]
		out_file = open(neigWord_file, 'w')
		print(data)
		for l in data:
			out_file.write((l[0]+'\t'+l[0]+":1.0\t")+'\t'.join(c for c in l[2:200])+ '\n')
		out_file.close()



"""
It computes the k nearest neighbors of a given word in the word2vec model, 
using the transportation profit function of (Guo et al,2016)
rij=(max(0,cos(wi,wj)))^(idf(wj)+b)
g_idf=idf+b
"""
def knn(word,g_idf,word2vec,k):
	# compute rij
	# make a sorted insert: 
	# go throgh the list and find the corresponding index of the desired element
	# make list.insert(index, obj) with: obj=(wordj,rij)
	neighbors=[]
	for w_i in word2vec.vocab:
		sim = 0.0
		try:
			sim = max(0,(1.0 - spatial.distance.cosine(word2vec[w_i], word2vec[word])))
		except: 
			sim = 0.0
		sorted_insert(w_i,sim**g_idf,neighbors)
	return (neighbors[:k])

"""
It makes a sorted insert of a tuple of values in a list.
"""
def sorted_insert(w_i,sim,neighbors):
	aux = neighbors
	i = 0
	while i<len(aux):
		if aux[i][1]<=sim:
			aux[i:i]=[(w_i,sim)]
			return(aux)
		else:
			i+=1
	
	if i==len(aux):
		return(aux.append((w_i,sim)))


"""
Makes a descendent sorting of alist in input 
"""
def insertionSort(alist):
	for index in range(1,len(alist)):
		currentvalue = alist[index]
		position = index
		while position>0 and alist[position-1]<currentvalue:
			alist[position]=alist[position-1]
			position = position-1
		alist[position]=currentvalue

"""
computes the dot-product similarity between two vectors, a and b, with alpha power 
"""

def sim(a,b,alpha):
  res = numpy.dot(a,b)**alpha
  return 0.0 if math.isnan(res) else res