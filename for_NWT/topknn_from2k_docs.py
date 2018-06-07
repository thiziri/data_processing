import heapq
import gensim
import numpy
from scipy import spatial
import math
from utils import cleanWord
import pyndri

def sim(a,b):
	res = max(numpy.dot(a,b),0)
	return 0.0 if math.isnan(res) else res

def sim_cos(a,b,alpha):
	return max(sim(a,b),0.0)**alpha

#embeddings = "/home/thiziri/Documents/monTravail/DOCTORAT/COLLECTIONS_TOPICS/MES_COLLECTIONS/word2vec_Models/robust/Rob04_wordEmbedding_dim300_win10_minCount5.txt"
embeddings = '/users/iris/tbelkace/mesTests/GoogleNews-vectors-negative300.bin.gz'

qts=[x.strip() for x in open("/users/iris/tbelkace/TOPICS/gov/gov_queryterms").readlines()]

idf={}
for line in open("/users/iris/tbelkace/Project/2ndYear/prepared_data/document_terms/GOV2wv.idf.txt").readlines():
	parts=line.strip().split("\t")
	idf[parts[0]]=float(parts[1])

model = gensim.models.Word2Vec.load_word2vec_format(embeddings, binary=True)
vocab = set(model.vocab.keys())

#del model


google2clean={} #dictionary of cleaned words
"""
for w in vocab:
	wn=w
	if not w in idf:
		wn=cleanWord(w,"krovetz")
	google2clean[w]=wn

#save the cleaned words
numpy.save('/users/iris/tbelkace/Project/2ndYear/2ndYear/data_preparation/for_NWT/generated/google2clean_gov.npy',google2clean)
"""
#load with: 
google2clean=numpy.load('/users/iris/tbelkace/Project/2ndYear/2ndYear/data_preparation/for_NWT/generated/google2clean.npy').item()
#print(google2clean) # display the dictionnary

run2k = "/users/iris/tbelkace/RUNS/INDRI_RUNS/2k_runs/tfidf_gov2_indri2k"
docs_2k = set([l.strip().split(" ")[2] for l in open(run2k,'r').readlines()])

docset_coll = "/users/iris/tbelkace/Project/2ndYear/prepared_data/document_terms/docset_GOV2"
vocab_run2k = set()
with open(docset_coll,'r') as inFile:
	for l in inFile:
		if l.strip().split("\t")[0] in docs_2k:
			doc = set()
			doc = set([c.split(":")[0] for c in l.strip().split("\t")[2:]])
			vocab_run2k |= doc

#localvocab = vocab & set(idf.keys())
localvocab = (vocab | set(google2clean.values())) & (set(qts)|set(idf.keys())) 
#localvocab_2k = localvocab & vocab_run2k

#del vocab_run2k
#del localvocab

normmodel = {}
"""
for t in vocab:
	if google2clean[t] in normmodel:
		normmodel[google2clean[t]]+=model[t]/numpy.linalg.norm(model[t])
	elif google2clean[t] in localvocab:#(set(qts)|localvocab):
		normmodel[google2clean[t]]=model[t]/numpy.linalg.norm(model[t])

del google2clean
#save the cleaned words vectors normalized
numpy.save('/users/iris/tbelkace/Project/2ndYear/2ndYear/data_preparation/for_NWT/generated/normModel_gov.npy',normmodel)
"""
normmodel = numpy.load('/users/iris/tbelkace/Project/2ndYear/2ndYear/data_preparation/for_NWT/generated/normModel.npy').item()
localvocab_2k = set(normmodel.keys()) & vocab_run2k
del vocab_run2k
del localvocab

del model

capacity = 200
for qt in set(qts):
	topn = []
	if qt in normmodel:
		for t in localvocab_2k-set([qt]):
			try:
				idf_qt = idf[qt]
			except:
				idf_qt = 0
			s = sim(normmodel[qt],normmodel[t])**2
			if len(topn) < capacity:
				heapq.heappush(topn, (s,t))
			else:
				x = heapq.heappop(topn)
				heapq.heappush(topn, (s,t) if x[0] < s else x)

	if topn!=[]:
		print("\t".join([qt,qt+":1.0"]+[x[1]+":"+str(x[0]) for x in sorted(topn,reverse=True)]))
	else:
		print("\t".join([qt,qt+":1.0"]))

