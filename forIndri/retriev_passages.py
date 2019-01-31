# ref: https://www.datacamp.com/community/tutorials/python-xml-elementtree

import xml.etree.ElementTree as ET
import sys
import json
import logging
import os
from os.path import join
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ").replace("<?xml version=\"1.0\" ?>\n", "")


logging.basicConfig(filename="logs/parse_xml_parameters4indri.log", level=logging.DEBUG)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = json.load(open(config_file))
    logging.info('Config: '+json.dumps(config, indent=2))
    tree = ET.parse(config["parameters"])
    root = tree.getroot()
    out = config["output"]

    print("Wait while retrieving passages for different queries ...")
    temp_parameters = join(config["output"], "temp_Q.xml")
    for query in root.iter('query'):
        q_id, q_txt = query.findtext("number"), query.findtext("text").strip()
        print(q_id, q_txt)
        c_root = ET.Element("parameters")
        c_query = ET.SubElement(c_root, "query")
        type_ = ET.SubElement(c_query, "type")
        type_.text = "indri"
        num = ET.SubElement(c_query, "number")
        num.text = q_id
        text = ET.SubElement(c_query, "text")
        text.text = "#combine[passage{pl}:{pw}]({q_txt})".format(pl=config["passage_length"],
                                                                 pw=config["sequence_length"], q_txt=q_txt)

        q_out = open(temp_parameters, 'w')
        q_out.write(prettify(c_root))
        q_out.close()

        os.chdir(config["indri"])
        passages = os.popen("./runquery/IndriRunQuery {param} -count={c} -index={i}".format(param=temp_parameters,
                                                                                       c=config["count"],
                                                                                       i=config["index"])).read()
        out_passages = open(join(config["output"], q_id), 'w')
        out_passages.write(passages)

    os.remove(temp_parameters)
    print("Finished.")
