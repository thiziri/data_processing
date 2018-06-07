import pyndri
from os.path import join
import os.path
import os, codecs
import docopt
from tqdm import tqdm
import collections
import sys
sys.path.append("/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/matchzoo/inputs")
sys.path.append("/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/matchzoo/utils")
from preprocess import *
from preparation import *
import six

"""
Read the relation file.
Return: list of tuples (label, q_id, doc_id)
"""
def get_relations(file_path):
    rels = []
    f = codecs.open(file_path, 'r', encoding='utf8')
    for line in tqdm(f):
        line = line.strip()
        label, id1, id2 = line.split()
        rels.append((label, id1, id2))
    return rels

if __name__ == '__main__':
	args = docopt.docopt("""
		Usage:
			prepare_data.py --d=<data> 

		Example:
			prepare_data.py --d=/home/thiziri/Documents/DOCTORAT/SOFT/MatchZoo/my_tests/custom_test/data/AP88/from_qrels

		Options:
			--d=<data>	Path to MatchZoo data corpus.txt and relation.txt.
		""")
	
	basedir = args["--d"]

	print("Prerpocess corpus file ...")
	

	preprocessor = Preprocess(word_seg_config = { 'enable': True, 'lang': 'en' },
		doc_filter_config = { 'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE },
		word_stem_config = { 'enable': False },
		word_lower_config = { 'enable': False },
		word_filter_config = { 'enable': True, 'stop_words': [], 'min_freq': 1, 'max_freq': six.MAXSIZE, 'words_useless': None })
	dids, docs = preprocessor.run(join(basedir, 'corpus.txt'))
	preprocessor.save_word_dict(join(basedir, 'word_dict.txt'))
	preprocessor.save_words_stats(join(basedir, 'word_stats.txt'))

	fout = open(join(basedir, 'corpus_preprocessed.txt'), 'w')
	for inum, did in enumerate(dids):
		fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
	fout.close()
	print('Preprocess finished.')
	del preprocessor
	
	


	print("Relation files preparation ...")
	prepare = Preparation()
	rels = get_relations(join(basedir, 'relation.txt'))
	rel_train, rel_valid, rel_test = prepare.split_train_valid_test_for_ranking(rels, [0.6, 0.2, 0.2])#split_train_valid_test(rels, (0.8, 0.1, 0.1))
	prepare.save_relation(join(basedir, 'relation_train.txt'), rel_train)
	prepare.save_relation(join(basedir, 'relation_valid.txt'), rel_valid)
	prepare.save_relation(join(basedir, 'relation_test.txt'), rel_test)
	print("Relation process finished.")
	
	

	print('Done.')