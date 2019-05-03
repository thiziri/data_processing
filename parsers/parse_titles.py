# coding: utf-8

# Parse every document title in the TREC dataset.
# tested in GOV2 dataset

import sys
import json
import os
import gzip
from tqdm import tqdm
from os.path import join


def parse(if_gzip, file):
    #if if_gzip:
    try:
        sample = gzip.open(file,
                           'rt',
                           encoding="ISO-8859-1")
        try:
            data = str(sample.read())
        except OSError:
            file = file.split('.')[0]
            os.system("uncompress " + file)
            data = open(file, 'r', encoding="ISO-8859-1").read()
    except:
        try:
            data = open(file, 'r').read()
        except:
            data = open(file, 'r', encoding="ISO-8859-1").read()
    return data


if __name__ == '__main__':

    # read config file:
    config = json.load(open(sys.argv[1]))

    to_read_beg = "<" + config["read"] + ">"
    to_read_end = "</" + config["read"] + ">"
    to_write_beg = "<" + config["write"] + ">"
    to_write_end = "</" + config["write"] + ">"

    print("Wait while arsing documents ...")
    num = 0
    for fold in tqdm(os.listdir(config["data_set"])):
        print(fold)
        docs = []
        if "readme" not in fold.lower():
            if os.path.isfile(os.path.join(config["data_set"], fold)):
                data = parse(config["if_gzip"], os.path.join(config["data_set"], fold))
                docs = docs + data.lower().split("</doc>")

            else:
                for sub_fold in os.listdir(os.path.join(config["data_set"], fold)):
                    data = parse(config["if_gzip"], os.path.join(os.path.join(config["data_set"], fold), sub_fold))
                    docs = docs + data.lower().split("</doc>")

            # parse titles with split : works

            print("{} documents to parse \n".format(len(docs)))
            # print(docs[0])
            doc_titles = {}
            for doc in tqdm(docs):
                    doc_id = doc.split('<docno>')[-1].split('</docno>')[0].upper().strip()
                    doc_title = doc.split(to_read_beg)[1].split(to_read_end)[0] if to_read_beg in doc else ''
                    doc_titles[doc_id] = doc_title

            # verifications

            print("\n{} parsed documents.\n".format(len(doc_titles)))
            # print(list(doc_titles.keys())[0], '\t', doc_titles[list(doc_titles.keys())[0]])
            empty = [d for d in doc_titles if len(doc_titles[d].split()) < 1]  # count empty documents
            print("{} NO title documents".format(len(empty)))
            print(empty)

            # filter according to the files list:
            documents_ok = [l.strip() for l in open(config["docs_list"], 'r').readlines()] \
                if bool(config["docs_list"]) else list(doc_titles.keys())

            # writing to TREC like dataset
            out = config["output"]  # "title_docs"
            if not os.path.exists(out):
                os.mkdir(out)
            with open(join(out, fold.split('.')[0]+".txt"), 'w') as trec:
                for d in tqdm(doc_titles):
                    if d != "" and d in documents_ok:
                        doc_schem = """<DOC>
<DOCNO>doc_id</DOCNO>
<TITLE>
doc_title
</TITLE>
</DOC>
"""  # the document skeleton
                        doc_schem.replace("<TITLE>", to_write_beg).replace("</TITLE>", to_write_end)
                        doc = doc_schem.replace("doc_id", d).replace("doc_title", doc_titles[d])
                        trec.write(doc)


