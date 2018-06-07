#construcrQueryFileForIndri

import collections
from collections import defaultdict
import operator
from os import listdir 
from os.path import join
import docopt
from pyndri import escape
import nltk
from nltk import tokenize
import re
from nltk.tokenize.moses import MosesTokenizer

def extractTopics(pathTop):
	print("Extraction de : %s" %(pathTop))
	nb=0
	topics={}

	for f in listdir(pathTop):
		f = open(join(pathTop,f), 'r')   # Reading file
		l = f.readline().lower()
		# extraction des topics
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


if __name__ == "__main__":
	print("\n------Begin------\n")
	args = docopt.docopt("""
	    Usage:
	        construcrQueryFile.py <topics_folder> <collection_name> <outputfolder> 
	        
	    """)
	
	print("\nBeging, parameters : \n")
	print(args)

	topics = extractTopics(args["<topics_folder>"])

	outputFile = open(join(args["<outputfolder>"],"RetrievalParameterFile_{name}.xml".format(name=args["<collection_name>"])), 'w')
	outputFile.write("<parameters>\n")

	tokenizer=MosesTokenizer()

	prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
	tops = {}
	for top in topics:
		terms=topics[top].split()
		toptext=""
		for t in terms:
			if (prog.match(t)):
				t=t.replace('.','')
				toptext=toptext+" "+t
		toptext=escape(toptext)
		tops[top]=tokenizer.tokenize(toptext,return_str=True)

	topics = collections.OrderedDict(sorted(tops.items()))

	for t in topics :
		print("topic : {t}".format(t=t))
		outputFile.write(" <query>\n  <type>indri</type>\n")
		outputFile.write("  <number>{num}</number>\n".format(num=int(t)))
		outputFile.write("  <text>\n")
		outputFile.write("   {txt}\n".format(txt=topics[t]))
		outputFile.write("  </text>\n")
		outputFile.write(" </query>\n")

	outputFile.write("</parameters>")
	print("\nEnded.")





