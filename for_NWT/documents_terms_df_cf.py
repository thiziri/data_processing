# Index reading to prepare a <term 	df 	tf> file of the index content
import logging, sys, pprint
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO) 
from os.path import join
import sys
import docopt
import pyndri
import collections
from collections import defaultdict
import operator
from utils import indexToDfCf_file, indexToDocSet_file

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        documents_terms_df_cf.py <index_path> <outputfolder> <collection_name> --run=<path_to_run_files>
	    
	    Options:
	        --run=<path_to_run_files>    Provides run files if want to get content of specific documents from a trec run file

	        """)

	print("\nIndex reading ... \n")
	index = pyndri.Index(args["<index_path>"])

	out1=open(join(args["<outputfolder>"], "terms_df_cf_"+args["<collection_name>"]),"w")
	indexToDfCf_file(index,out1) # reading terms

	out2=open(join(args["<outputfolder>"], "docset_"+args["<collection_name>"]),"w") 
	indexToDocSet_file(index,out2,args["--run"])


	print("Index reading finished.")
