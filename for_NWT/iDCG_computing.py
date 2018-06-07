# computes the NDCG of a given run with a corresponding qrels (TREC format)
import logging, sys, pprint
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO) 
from os.path import join
import sys
import docopt
import collections
from collections import defaultdict
import operator
import math

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        iDCG_computing.py <collection_name> --out=<path_to_output_file> [--run=<run_file>|--qrels=<qrels_file>]
	    
	    Options:
	        --out=<path_to_output_file>    Provides an output file, if doesn't exist will be created
	        --run=<run_file>    Provide a run file within TREC format
	        --qrels=<qrels_file>    Provide ground truth file within TREC format

	        """)

	iDCG={}

	if (bool(args["--run"])):
		run=open(args["--run"],"r")
		lines=run.readlines()

		query_lines={}
		for l in lines:
			q=int(l.split()[:1][0]) # get the query ID
			if q in query_lines:
				query_lines[q].append(l.split()[1:])
			else:
				query_lines[q]=[]
				query_lines[q].append(l.split()[1:])

		# exemple result:
		# query_lines={304: [['Q0', 'FBIS4-67703', '1', '-6.91998700', 'galago'], ['Q0', 'FBIS4-67703', '1', '-6.91998700', 'galago'], ['Q0', 'FBIS4-67704', '1', '-6.91998700', 'galago']], 302: [['Q0', 'FBIS4-67701', '1', '-6.91998700', 'galago'], ['Q0', 'FBIS4-67701', '1', '-6.91998700', 'galago']], 303: [['Q0', 'FBIS4-67702', '1', '-6.91998700', 'galago'], ['Q0', 'FBIS4-67702', '1', '-6.91998700', 'galago']]}

		for q in query_lines:
			iDCG[q]=0.0
			lines=query_lines[q]
			i=0
			for l in lines:
				i+=1
				#iDCG[q]+=((2**float(l[3]))-1)/(math.log2(i+1))
				iDCG[q]+=(float(l[3]))/(math.log2(i+1))
				#if(i==1000): break
		#print(iDCG)

	else:
		run=open(args["--qrels"],"r")
		lines=run.readlines()

		query_lines={}
		for l in lines:
			q=int(l.split()[:1][0]) # get the query ID
			if q in query_lines:
				query_lines[q].append(l.split()[1:])
			else:
				query_lines[q]=[]
				query_lines[q].append(l.split()[1:])

		# exemple result:
		# query_lines={301: [['0', 'FBIS4-67703', '1'], ['0', 'FBIS3-10082','1']], 302: [['0', 'FBIS3-67701', '1']]}

		#print(query_lines)

		for q in query_lines:
			iDCG[q]=0.0#int(query_lines[q][0][2])
			lines=query_lines[q][1:]
			rel=[]
			rel2=[]
			for l in lines:
				rel.append(int(l[2]))
				rel2=sorted(rel,reverse=True)
			#iDCG[q]=rel2[0]+sum([((2**relScore)-1)/(math.log2(rank)) for rank, relScore in enumerate(rel2[1:], start=2)])
			iDCG[q]=rel2[0]+sum([relScore/(math.log2(rank+1)) for rank, relScore in enumerate(rel2[1:], start=1)])
			#if(i==1000): continue
		#print(iDCG)



	out=open(join(args["--out"],"iDCG"+args["<collection_name>"]),"w")
	for q in iDCG:
		out.write("{q}\t{idcg}\n".format(q=q,idcg=iDCG[q]))

	print("Finished.\nResults on: "+args["--out"])