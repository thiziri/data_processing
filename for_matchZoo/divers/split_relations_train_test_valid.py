import codecs
from tqdm import tqdm
import hashlib


def run_with_one_corpus_trec(file_path, corpus_file):
        hashid = {}
        f = codecs.open(corpus_file, 'r', encoding='utf8')
        print("Getting TREC ids ...")
        for line in tqdm(f):
            _id = line.strip().split()[0]
            text = line.strip()[len(_id):].strip()
            hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
            hex_dig = hash_obj.hexdigest()
            hashid[hex_dig] = _id
        # print(hashid)
        corpus = {}
        rels = []
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in f:
            line = line
            line = line.strip()
            label, t1, t2 = parse_line(line)
            id1 = get_text_id_trec(hashid, t1)
            id2 = get_text_id_trec(hashid, t2)
            corpus[id1] = t1
            corpus[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus, rels


def get_text_id_trec(hashid, text):
        hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
        hex_dig = hash_obj.hexdigest()
        try:
            return hashid[hex_dig]
        except:
            print("text not found : "+text)
            exit()


def parse_line(line, delimiter='\t'):
        subs = line.split(delimiter)
        # print('subs: ', len(subs))
        if 3 != len(subs):
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]


corpus, rels = run_with_one_corpus_trec('/home/thiziri/Documents/DOCTORAT/osirim_data/projets/iris/PROJETS/WEIR/code/2ndYear/MatchZoo/my_tests/data/AP88/from_qrels/sample.txt' , 
    '/home/thiziri/Documents/DOCTORAT/osirim_data/projets/iris/PROJETS/WEIR/code/2ndYear/MatchZoo/my_tests/custom_test/data/AP88/from_qrels/corpus.txt' )
print(corpus, rels)