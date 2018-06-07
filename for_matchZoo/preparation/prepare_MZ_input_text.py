import pyndri
from os.path import join
import os
import codecs
import sys
import json
import logging
from tqdm import tqdm
from collections import defaultdict

sys.path.append('../for_matchZoo/utils')

from tools4text import extractTopics, clean, get_qrels, save_corpus, get_docs_from_run, run2relations
from tools4text import rank_to_relevance, path_leaf, remove_extension


logging.basicConfig(filename='collect2MZinpuText.log',level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    #print(sys.argv[1])
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))

    print("Data extraction\nConfiguration: ")
    print(json.dumps(config, indent=2), end='\n')

    print("Reading index ...")
    index = pyndri.Index(config["indexed_data"])
    _, id2token, _ = index.get_dictionary()
    externalDocId = {}
    for doc in range(index.document_base(), index.maximum_document()):
        extD, _ = index.document(doc)
        externalDocId[extD] = doc
    queries = extractTopics(config["queries"])
    queries_text = {}
    q_times = defaultdict(int)
    for q in queries:
        q_text = clean(queries[q], "krovetz",{})
        q_times[q_text] += 1
        queries_text[q] = q_text if q_times[q_text]==1 else ' '.join([q_text, str(q_times[q_text])])

    out_trec_f = join(config["output_folder"], "trec_corpus.txt")
    out_t = codecs.open(out_trec_f,"w",encoding='utf8')

    print("Collection2Text ...")
    nl = 0
    relations = []

    if config["from_qrels"]:
        qrels = get_qrels(config["relevance_judgements"])  # qrels[(q, doc)] = rel q:str, rel:int
        ranked_documents = set([e[1] for e in qrels])
        if bool(config["run_file"]):
            ranked_documents = ranked_documents.union(get_docs_from_run(config["run_file"]))
        print("totalling: %d documents" % len(ranked_documents))
        nl = save_corpus(queries_text, ranked_documents, index, id2token, externalDocId, out_t)

        logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")

        relations = [((e[0], e[1]), qrels[e]) for e in qrels]  # same content (q, doc):rel, q:int
        logging.info('From relevance judgements : ' + config["relevance_judgements"])

    elif config["from_run"]:
        logging.info("From run: " + config["run_file"])
        qrels = get_qrels(config["relevance_judgements"]) if bool(config["relevance_judgements"]) else []

        ranked_documents = get_docs_from_run(config["run_file"])
        if bool(config["relevance_judgements"]):
            ranked_documents = ranked_documents.union(set([e[1] for e in qrels]))
            
        print("totalling: %d documents" % len(ranked_documents))
        nl = save_corpus(queries_text, ranked_documents, index, id2token, externalDocId, out_t)
        logging.info("Corpus file saved to " + out_trec_f+" with "+str(nl)+" lines")
        relations = run2relations(config["run_file"], config["binary_judgements"], qrels, config["scales"], config["ranks"])

    # write corpus content
    if not config["cross_validation"]:
        nl = 0
        logging.info("Without cross-validation, resulting files are in " + config["output_folder"])
        out_f = join(config["output_folder"], "sample.txt")
        out = codecs.open(out_f, "w", encoding='utf8')
        for e in relations:
            q = e[0][0]
            doc = e[0][1]
            rel = e[1]
            try:
                doc_text = " ".join([id2token[x] for x in index.document(externalDocId[doc])[1] if x != 0])
            except:
                doc_text = ""
            if doc_text != "":
                out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
                nl += 1
        print("Collection2Text finished.\nResults in {f}\n{n} lines.".format(f=out_f, n=nl))

    else:
        # perform n_cross validation
        logging.info("Data cross-validation, resulting files are in %s" .format(config["output_folder"]))

        def select_rel_by_qids(qid_list, relations_list):
            # select relations 
            return set([re for re in relations_list if re[0][0] in qid_list])


        folds = {}
        qrels = get_qrels(config["relevance_judgements"]) if bool(config["relevance_judgements"]) else []
        # qrels[(str,str)]:int
        relations_test = []
        if bool(config["run_file"]):
            relations_test = run2relations(config["run_file"],
                                       config["binary_judgements"],
                                       qrels,
                                       config["scales"],
                                       config["ranks"],
                                       config["k"])

        for fold in os.listdir(config["split_data"]):
            # print("fold ", fold)
            qid_test = [l.strip() for l in open(join(join(config["split_data"], fold), "test_.txt"), "r").readlines()]
            qid_valid = [l.strip() for l in open(join(join(config["split_data"], fold), "valid_.txt"), "r").readlines()]
            qid_train = [l.strip() for l in open(join(join(config["split_data"], fold), "train_.txt"), "r").readlines()]

            rel_train = select_rel_by_qids(qid_train, relations)
            rel_valid = select_rel_by_qids(qid_valid, relations)
            rel_test = set()
            if not config["reranking"]:
                rel_test = select_rel_by_qids(qid_test, relations)
            else:
                # print(relations_test)
                rel_test = select_rel_by_qids(qid_test, relations_test)
                # print(qid_test)
                # print(rel_test)
            folds[path_leaf(fold)] = {"test": rel_test, "valid": rel_valid, "train": rel_train}

        print("save relation files in different folds ...")
        for fold in tqdm(folds):
            f = join(config["output_folder"], fold)
            os.mkdir(f)
            for group in folds[fold]:
                out = open(join(f, "corpus_"+group+".txt"), "w")
                for r in tqdm(folds[fold][group]):
                    q = r[0][0]
                    doc = r[0][1]
                    rel = r[1]
                    try:
                        doc_text = " ".join([id2token[x] for x in index.document(externalDocId[doc])[1] if x != 0])
                    except:
                        doc_text = ""
                    if doc_text != "":
                        out.write("{r}\t{q}\t{d}\n".format(r=rel, q=queries_text[q], d=doc_text, encoding='utf8'))
                out.close()
