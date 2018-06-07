
from os.path import join
import os.path
import os
import docopt
from tqdm import tqdm
import collections
from random import shuffle
import itertools

def read_relations(file):
	relations = {}
	with open(file, "r") as f:
		for l in tqdm(f):
			rel = l.strip().split()[0]
			q = l.strip().split()[1]
			d = l.strip().split()[2]
			if q in relations:
				relations[q][d] = rel
			else:
				relations[q] = {}
				relations[q][d] = rel
	return (relations)

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq


def split_train_valid_test(relations, ratio=(0.8, 0.1, 0.1)):
        shuffle(relations)
        total_rel = len(relations)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid
        rel_train = relations[: num_train]
        rel_valid = relations[num_train: valid_end]
        rel_test = relations[valid_end:]
        return rel_train, rel_valid, rel_test

def split_train_valid_test_for_ranking(relations, ratio=(0.8, 0.1, 0.1)):
        qid_group = set()
        for r, q, d in relations:
            qid_group.add(q)
        qid_group = list(qid_group)

        random.shuffle(qid_group)
        total_rel = len(qid_group)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid

        qid_train = qid_group[: num_train]
        qid_valid = qid_group[num_train: valid_end]
        qid_test = qid_group[valid_end:]

        def select_rel_by_qids(qids):
            rels = []
            qids = set(qids)
            for r, q, d in relations:
                if q in qids:
                    rels.append((r, q, d))
            return rels

        rel_train = select_rel_by_qids(qid_train)
        rel_valid = select_rel_by_qids(qid_valid)
        rel_test = select_rel_by_qids(qid_test)

        return rel_train, rel_valid, rel_test

if __name__ == '__main__':
	args = docopt.docopt("""
		Usage:
		    relation2crossFolds.py --r=<relation_file> --o=<output_folder> --f=<folds_number>

		Example:
		    relation2crossFolds.py --r='/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/my_tests/custom_test/data/AP88/from_indri_runs/lm/ranking_binary_judgement/relation.txt' --o='/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/my_tests/custom_test/data/AP88/from_indri_runs/lm/ranking_binary_judgement/folds' --f=5

		Options:
		    --r=<relation_file>	Give constructed relation.txt file. 
		    --o=<output_folder>	Give the output folder where constructed folds will be stored.
		    --f=<folds_number>	Give the number of train/test/validation folds.

		""")

	print("Reading relations")
	relations = read_relations(args["--r"])
	queries = list(relations.keys())
	shuffle(queries) # shuffling queries
	folds = split_seq(queries, int(args["--f"])) # split into 5-folds for cross validation
	#print(folds)
	print("Saving folds...")
	for idx,f in tqdm(enumerate(folds)):
		#print(f)
		test = f
		valid = folds[idx+1] if idx<len(folds)-1 else folds[0]
		train = list(itertools.chain.from_iterable([e for e in folds if e not in [test, valid]]))
		#print("test{idx} {t} valid{idx} {v} train{idx} {tr}".format(t=test, v=valid, tr=train, idx=idx+1))
		fold = join(args["--o"], "fold_"+str(idx))
		if not os.path.exists(fold):
			os.makedirs(fold)
		tr = join(fold, "relation_train.txt")
		tst = join(fold, "relation_test.txt")
		vld = join(fold, "relation_valid.txt")
		# write relation files
		with open(tr, "w") as out:
			for q in train:
				for d in relations[q]:
					line = "{rel} {q} {d}\n".format(rel=relations[q][d], q=q, d=d)
					out.write(line)
		with open(tst, "w") as out:
			for q in test:
				for d in relations[q]:
					line = "{rel} {q} {d}\n".format(rel=relations[q][d], q=q, d=d)
					out.write(line)
		with open(vld, "w") as out:
			for q in valid:
				for d in relations[q]:
					line = "{rel} {q} {d}\n".format(rel=relations[q][d], q=q, d=d)
					out.write(line)


