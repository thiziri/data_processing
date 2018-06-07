
import os
import sys
import math

infile = "qrels.rob04.txt"
outfile = "rob04.qrels.idcg_at20"

#fout = open("rob04.qrels.idcg")
qinfo = {}
with open(infile,"r") as f:
    for line in f:
        r = line.split()
        qid = int(r[0])
        did = r[2]
        clabel = int(r[3])
        if qid not in qinfo:
            qinfo[qid] = {}
        qinfo[qid][did] = clabel
fout = open(outfile,"w")
for qid,dlist in qinfo.items():
    nlist = sorted(dlist.items(),key=lambda d:d[1],reverse = True)
    inum = 0
    idcg = 0.0;
    for d in nlist:
        idcg += float(2**d[1] - 1) / float(math.log(inum+2))
        inum += 1
        if inum >= 20:
            break
    fout.write("%d\t%f\n"%(qid,idcg))
fout.close()


