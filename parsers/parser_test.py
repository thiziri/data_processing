
# coding: utf-8

# In[29]:


# needed libraries

import xml.etree.ElementTree as ET
import gzip
from tqdm import tqdm


# In[30]:


# unzip and read

sample = gzip.open("docs/00.gz", 'rt', encoding = "ISO-8859-1")
data = str(sample.read())
xml_data = "<collection>" + data + "</collection>"
print(xml_data[:100])


# In[34]:


# parse titles with split : wooooooooooooooooooooooooooooorks

docs = data.lower().split("</doc>")
print("{} documents to parse".format(len(docs)))
#print(docs[0])
doc_titles = {}
for doc in tqdm(docs):
    doc_id = doc.split('<docno>')[-1].split('</docno>')[0].upper()
    doc_title = doc.split('<title>')[-1].split('</title>')[0] if '<title>' in doc else ''
    doc_titles[doc_id] = doc_title


# In[35]:


# verifications

print("{} parsed documents.\n".format(len(doc_titles)))
print(list(doc_titles.keys())[0], '\t', doc_titles[list(doc_titles.keys())[0]])
# empty = len([d if len(doc_title[d].split(' '))==0 for d in doc_titles])  # count empty documents
empty = [d for d in doc_titles if len(doc_titles[d].split())<1]
print("{} NO title documents".format(len(empty)))
print(empty)


# In[36]:


# writing to TREC like dataset

from os.path import join

out = "title_docs"
with open(join(out, "00.txt"), 'w') as trec:
    for d in tqdm(doc_titles):
        doc_schem = """<DOC>
<DOCNO>doc_id</DOCNO>
<TITLE>
doc_title
</TITLE>
</DOC>
"""
        doc = doc_schem.replace("doc_id", d).replace("doc_title", doc_titles[d])
        trec.write(doc)
        

