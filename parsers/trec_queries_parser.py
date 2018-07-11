# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""
Trec like parser"""

import re
import nltk
import ntpath
import sys
import json
from collections import defaultdict
from tqdm import tqdm
from os import listdir
from os.path import join
from nltk.stem.porter import PorterStemmer
from krovetzstemmer import Stemmer
from bs4 import BeautifulSoup as Soup
from nltk.corpus import stopwords


"""
It return the file name extracted from a path
"""
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


"""
removes the file extension:
example: file.txt becomes file
return: file name without extension
"""

"""
Cleans the input text of special characters
return cleaned text
"""
def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('^'): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('!'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('€'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('#'): ' ',
        ord('_'): ' ',
        ord('@'): ' ',
        ord('~'): ' ',
        ord('='): None,
        ord('*'): None,
    })


"""
Performs stemming according to the selected algo
return stemed text
"""
def stem(algo, text):
    if algo == "krovetz":
        stemmer = Stemmer()
        return stemmer.stem(text)
    elif algo == "porter":
        stm = PorterStemmer()
        return stm.stem(text)
    print("ERROR STEMMING: {t} unkown.".format(t=algo))


"""
Performs cleaning and stemming 
return cleaned and stemmed text
"""
def clean(text_to_clean, steming, stoplist):
    prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
    tex = []
    for w in text_to_clean.split():
        if prog.match(w):
            w = w.replace('.', '')
        tex.append(w)
    text = " ".join(tex)
    text = ' '.join(escape(text).split())
    text = " ".join(nltk.word_tokenize(text))
    text = " ".join([stem(steming, w) for w in text.split() if w not in stoplist])
    return text


""" 
Extract TREC topics on the pathTop parameter as dictionnary. 
return dictionnary of queries.
ex: {0:"this is a text of the topic"}
"""
def extract_topics_body(path_top, indices):
    print("Extraction de : %s" % path_top)
    topics = {}
    for f in listdir(path_top):
        file = join(path_top, f)  # Reading file
        print(file)
        handler = open(file).read()
        soup = Soup(handler, "lxml")
        for topic in soup.findAll('top'):
            if topic is not None:
                top_id, content = get_content(str(topic).lower(), indices)
                topics[top_id] = content
    return topics


"""
Parse a query content
return: tuple
"""
def get_content(topic, indices):
    # indices = {id: (get, ignore), value: (get, ignore)} exp: {"id": ("<num>", "number:"), "content": {"<title>": ""}}
    # ender = re.compile("\<\/?[a-z]+\>.*")
    starter = re.compile("\<[a-z]+\>.*")
    content_label = list(indices["content"].keys())
    content_label.append(indices["id"][0])
    # print(content_label)
    num = ""
    content = ""
    for line in topic.split('\n'):
        if line.strip() != "":
            if starter.match(line.strip().split()[0]) and line.strip().split()[0] in content_label:
                if line.strip().split()[0] == indices["id"][0]:
                    num = line.replace(indices["id"][0], "").replace(indices["id"][1], "").strip()
                else:
                    active = True
                    starts = line.strip().split()[0]
                    line = line.replace(starts, " ").replace(indices["content"][starts], " ")
                    content = " ".join([content + line.strip()])
            elif not starter.match(line.strip().split()[0]) and active:
                content = " ".join([content, line.strip()])
            else:
                active = False
    return num, content

if __name__ == '__main__':
    config_file = sys.argv[1]
    configure = json.load(open(config_file))
    config = configure["main_configuration"]
    topics = extract_topics_body(config["topics_folder"], config["indices"])
    print(json.dumps(config, indent=2), end='\n')

    print("Cleaning ...")
    stopWords = set(stopwords.words('english')) if config["stop"] else []
    topics_clean = {}
    q_times = defaultdict(int)
    for top in tqdm(topics):
        top_text = clean(topics[top], config["stemmer"], stopWords)
        q_times[top_text] += 1
        top_text if q_times[top] == 1 else ' '.join([top_text, str(q_times[top_text])])
        topics_clean[top] = top_text
        if config["stop"]:
            topics_clean[top] = " ".join([w for w in top_text.split()])

    out = open(config["output"]+'.'+config["output_format"], "w")
    if config["output_format"] == "txt":
        for top in topics_clean:
            out.write("{q}\t{txt}\n".format(q=top, txt=topics_clean[top]))
    elif config["output_format"] == "xml":
        out.write("<parameters>\n")
        for top in topics_clean:
            out.write(" <query>\n  <type>indri</type>\n")
            out.write("  <number>{num}</number>\n".format(num=int(top)))
            out.write("  <text>\n")
            out.write("   {txt}\n".format(txt=topics_clean[top]))
            out.write("  </text>\n")
            out.write(" </query>\n")

        out.write("</parameters>")
