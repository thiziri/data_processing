import heapq
import gensim
import numpy
import math

def sim(a,b,alpha):
  res = numpy.dot(a,b)**alpha
  return 0.0 if math.isnan(res) else res

qts=[x.strip() for x in open("/home/thiziri/Documents/monTravail/Logiciels/NWT_Guo_al16/myData/with_googleWE/query_terms").readlines()]

idf={}
for line in open("robustwv.idf.txt").readlines():
  parts=line.strip().split("\t")
  idf[parts[0]]=float(parts[1])

model = gensim.models.Word2Vec.load_word2vec_format('/home/thiziri/Documents/monTravail/DOCTORAT/COLLECTIONS_TOPICS/googleModel/GoogleNews-vectors-negative300.bin.gz', binary=True)
vocab = set(model.vocab.keys())
"""
google2clean={} #dictionary of cleaned words
for w in vocab:
	wn=w
	if not w in idf:
		wn=cleanWord(w)
	google2clean[w]=wn
"""

localvocab = vocab & set(idf.keys()) 
normmodel = {}
for t in (set(qts)|localvocab) & vocab:
  normmodel[t]=model[t]/numpy.linalg.norm(model[t])

del model

capacity = 200
for qt in set(qts) & vocab:
  topn = []
  for t in localvocab-set([qt]):
    s = sim(normmodel[qt],normmodel[t], idf[qt])
    if len(topn) < capacity:
      heapq.heappush(topn, (s,t))
    else:
      x = heapq.heappop(topn)
      heapq.heappush(topn, (s,t) if x[0] < s else x)
  print("\t".join([qt,qt+":1.0"]+[x[1]+":"+str(x[0]) for x in sorted(topn,reverse=True)]))

