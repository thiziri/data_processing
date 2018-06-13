# call the run2relation() method for powerful files
import sys
sys.path.append('../for_matchZoo/utils')

import logging
import json
from tqdm import tqdm
from os.path import join
from tools4text import get_qrels, rank_to_relevance, path_leaf

logging.basicConfig(filename='run2relations.log', level=logging.DEBUG)

K = 1000
QRELS = []
BINARY = False

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: ' + json.dumps(config, indent=2))

    print("Read run to relations\n", json.dumps(config, indent=2))
    try:
        K = config["k"]
    finally:
        pass
    try:
        QRELS = get_qrels(config["relevance_judgements"])
    except:
        pass
    try:
        BINARY = config["binary"]
    finally:
        pass

    f = join(config["output"], "run2relations_"+path_leaf(config["train_run"]))
    out = open(f, 'w')
    n = 0
    print("Wait while parsing run to relations ...")
    with open(config["train_run"], "r") as rank:
        i = 0
        j = -1
        queries_rank = []
        for line in tqdm(rank):
            if line is not None:
                j += 1
                q = str(int(line.strip().split()[0]))
                if q in queries_rank:
                    i += 1
                else:
                    queries_rank.append(q)
                    i = 1
                doc = line.strip().split()[2]
                if len(QRELS) == 0:
                    if not BINARY:
                        x = rank_to_relevance(i, config["scales"], config["ranks"])
                        rel = x if x is not None else 0  # multiscale relevance
                    else:
                        rel = 1 if i <= 10 else 0  # binary relevance
                else:
                    try:
                        rel = QRELS[(q, doc)]
                    finally:
                        rel = 0
                if i in range(K + 1):
                    out.write("{q}\tq0\t{d}\t{r}\n".format(q=q, d=doc, r=rel))
                    # print("{q}\tq0\t{d}\t{r}\n".format(q=q, d=doc, r=rel))
                    n += 1
    print("Finished.\nResults in {f}\n{n} lines".format(f=f, n=n))
