# construct a query file for runing INDRI

import collections
import docopt
import re
from nltk.tokenize.moses import MosesTokenizer
from tools4text import extract_trec_topics, extract_trec_million_queries, escape
from os.path import join
from tqdm import tqdm


if __name__ == "__main__":
    print("\n------Begin------\n")
    args = docopt.docopt("""
        Usage:
            construcrQueryFile.py <topics_folder> <collection_name> <outputfolder> [--tq | --mq]
            
        Options:
            --t    if topics_folder contains trec query files.
            --tm    if topics_folder contains trec_million query files.
        """)
    
    print("\nparameters : \n")
    print(args)

    topics = {}
    if args["--tq"]:
        topics = extract_trec_topics(args["<topics_folder>"])
    elif args["--mq"]:
        topics = extract_trec_million_queries(args["<topics_folder>"])
    else:
        print("No queries to extract")

    outputFile = open(join(args["<outputfolder>"],
                           "RetrievalParameterFile_{name}.xml".format(name=args["<collection_name>"])), 'w')
    outputFile.write("<parameters>\n")

    tokenizer = MosesTokenizer()

    prog = re.compile("[_\-\(]*([A-Z]\.(\ )*)*[_\-\(]*")
    tops = {}
    for top in topics:
        terms = topics[top].split()
        toptext = ""
        for t in terms:
            if prog.match(t):
                t = t.replace('.', '')
                toptext = toptext + " " + t
        toptext = escape(toptext)
        tops[top] = tokenizer.tokenize(toptext, return_str=True)

    topics = collections.OrderedDict(sorted(tops.items()))

    print("Writting .xml file ...")
    for t in tqdm(topics) :
        # print("topic : {t}".format(t=t))
        outputFile.write(" <query>\n  <type>indri</type>\n")
        outputFile.write("  <number>{num}</number>\n".format(num=int(t)))
        outputFile.write("  <text>\n")
        outputFile.write("   {txt}\n".format(txt=topics[t]))
        outputFile.write("  </text>\n")
        outputFile.write(" </query>\n")

    outputFile.write("</parameters>")
    print("\nEnded.")





