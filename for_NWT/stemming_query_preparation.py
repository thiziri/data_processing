#perform a stemming for input text
from utils import extractTopics
from utils import stem
from os.path import join
import os.path
from utils import escape
import re
import nltk
from nltk.corpus import stopwords
import docopt

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        stemming_query_preparation.py [--intxt=<text>|--inTopics=<path>] <outputfolder> [--stop] [--k|--p] --out=<prefix>
	    
	    Options:
	        --intxt=<text>    Provide path to text file or just text between cotes .
	        --inTopics=<path>    Provide path to TREC topics file .
	        --stop    Choose this option to perform stop words removel for the input text .
	        --k    Choose this option to use krovetz stemmer .
	        --p    Choose this option to use porter stemmer .
	        --out=<prefix>    Add a prefix for the output file .
	        
	    """)

	txt=""
	toProcess={}
	if bool (args["--intxt"]):
		txt="the input text"
		if os.path.isfile(args["--intxt"]):
			with open(args["--intxt"], 'r') as content_file:
				toProcess[0]= content_file.read()
		else:
			toProcess[0]= args["--intxt"]
	elif bool(args["--inTopics"]):
		txt=args["--inTopics"]
		toProcess=extractTopics(args["--inTopics"])
	#print(toProcess)
	algo=""
	if bool(args["--p"]):
		algo="porter"
	else:
		algo="krovetz"

	if bool (args["--stop"]):
		print("Stemming with {a} stemmer and stop words removel".format(a=algo))
	else:
		print("Stemming with {a} stemmer without stop words removel".format(a=algo))

	print("Please wait while we are stemming the text ... \n")


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

	if bool(args["--stop"]):
		stopWordsList=set(stopwords.words('english'))
		print(stopWordsList)
		for i in toProcess:
			listWords=[w for w in toProcess[i] if w not in stopWordsList]
			toProcess[i]=listWords
		#print(toProcess)

	#print(toProcess)	

	file=open(join(args["<outputfolder>"],algo+"_stemmed_"+args["--out"]),"w")

	#print(toProcess)

	for w in toProcess:
		content=str(w)
		for word in toProcess[w]:
			content+='\t'+word
		file.write(content+"\n")
			
	file.close()
