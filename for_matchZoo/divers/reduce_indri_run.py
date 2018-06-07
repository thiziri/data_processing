from os.path import join
import os.path
import os
import docopt
from tqdm import tqdm
from tools4text import path_leaf
from tools4text import docs_perQuery

if __name__ == '__main__':
	args = docopt.docopt("""
		Usage:
			reduce_indri_run.py [--qrels=<qrels_file>] [--run=<initial_run>] --indri_run=<indri_run> --out=<output_folder>

		Example:
			reduce_indri_run.py --run=<initial_run> --indri_run=<indri_run> --out=<output_folder>

		Options:
			--qrels=<qrels_file>	Give the qrels file if want to filter according to qrels.
			--run=<initial_run>	Provide the run that contains the concerned queries by test.
			--indri_run=<indri_run>	Provide the indri like run that will be compared to the initial run.
			--out=<output_folder>	Where the filtered run will be saved.
		""")
		
	print("Filtering run: ",args["--indri_run"])
	queries = set()

	if bool(args["--run"]):
		print("Reading queries from run: ", args["--run"])
		with open(args["--run"], "r") as run:
			for line in tqdm(run):
				if line != None:
					queries.add(int(line.strip().split()[0]))


	if bool(args["--run"]) and not (bool(args["--qrels"])):
		out = open(join(args["--out"], "reduced_"+path_leaf(args["--indri_run"])), "w")
		print("Filtering according to only run "+args["--run"])
		with open(args["--indri_run"], "r") as run:
			for line in tqdm(run):
				if line != None:
					if int(line.strip().split()[0]) in queries:
						out.write(line)

	elif (bool(args["--run"])) and (bool(args["--qrels"])):
		out = open(join(args["--out"], "qrels_reduced_"+path_leaf(args["--indri_run"])), "w")
		print("Filtering according to qrels "+args["--qrels"]+"and run "+args["--run"])
		qrels_docs = docs_perQuery(args["--qrels"])
		with open(args["--indri_run"], "r") as run:
			for line in tqdm(run):
				if line != None:
					if int(line.strip().split()[0]) in queries and line.strip().split()[2] in qrels_docs[line.strip().split()[0]]:
						out.write(line)
	else:
		out = open(join(args["--out"], "qrels_only_reduced_"+path_leaf(args["--indri_run"])), "w")
		print("Filtering according to qrels "+args["--qrels"])
		qrels_docs = docs_perQuery(args["--qrels"])
		with open(args["--indri_run"], "r") as run:
			for line in tqdm(run):
				if line != None:
					if line.strip().split()[0] in qrels_docs:
						if line.strip().split()[2] in qrels_docs[line.strip().split()[0]]:
							out.write(line)
	print("Finished.")