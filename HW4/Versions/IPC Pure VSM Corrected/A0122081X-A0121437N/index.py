#!/usr/bin/python
import re
import nltk
import sys
import getopt
import string

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from math import log10, sqrt, floor

import xml.etree.ElementTree as ET

stemmer = nltk.stem.porter.PorterStemmer()

stops = map(lambda x: stemmer.stem(x).lower(), stopwords.words('english'))
text_zones = ["Title", "Abstract"]
related_documents = ["Family Members", "Cites"]
ipc_class = ["IPC Class"]
ipc_subclass = ["IPC Subclass"]
ipc_group = ["IPC Group"]

def build_index(document_dir):
    """
    Builds the index.
    Returns a tuple with the total number of documents indexed and the index itself.
    """
    term_doc_id_index = {} # key: terms, value: [(doc_id, tf, norm_tf),...]
    doc_index = {} # key: doc_id, value: (num_tokens, [(term, tf),...])
    related_documents_index = {} # key: doc_id, value: [doc_id,...]

    ipc_classes = set()
    ipc_subclasses = set()
    ipc_groups = set()

    ipc_class_idx = {} # key: doc_id, value: ipc_class
    ipc_subclass_idx = {} # key: doc_id, value: ipc_subclass
    ipc_group_idx = {} # key: doc_id, value: ipc_group

    ipc_class_tokens = {} # key: ipc_class, value: {term: tf}
    ipc_subclass_tokens = {} # key: ipc_subclass, value: {term: tf}
    ipc_group_tokens = {} # key: ipc_group, value: {term: tf}

    term_ipc_class_idx = {} # key: term, value: {ipc_class: tf}
    term_ipc_subclass_idx = {} # key: term, value: {ipc_subclass: tf}
    term_ipc_group_idx = {} # key: term, value: {ipc_group: tf}

    files = listdir(document_dir)
    files = filter(lambda x: x.endswith('.xml'), files)
    files_set = set(map(lambda x: x[:-4], files))
    files.sort()

    for f in files:
        path = join(document_dir, f)
        if isfile(path):
            root = ET.parse(path).getroot() # XML File root
            doc_tokens = {} # key: term, value: tf
            doc_id = f[:-4] # Remove '.xml' from file name

            for child in root:
                attribute_name = child.attrib['name'].strip()
                if attribute_name in text_zones:
                    add_text_tokens_to_doc_tokens(child.text, doc_tokens)
                elif attribute_name in related_documents:
                    add_docs_to_related_documents_index(child.text, doc_id, files_set, related_documents_index)
                elif attribute_name in ipc_class:
                    ipc_class_idx[doc_id] = child.text.strip()
                    ipc_classes = ipc_classes | set([child.text.strip()])
                elif attribute_name in ipc_subclass:
                    ipc_subclass_idx[doc_id] = child.text.strip()
                    ipc_subclasses = ipc_subclasses | set([child.text.strip()])
                elif attribute_name in ipc_group:
                    ipc_group_idx[doc_id] = child.text.strip()
                    ipc_groups = ipc_groups | set([child.text.strip()])

            if doc_id in ipc_class_idx:
                add_doc_tokens_to_ipc_tokens(ipc_class_idx[doc_id], doc_tokens, ipc_class_tokens)
            if doc_id in ipc_subclass_idx:
                add_doc_tokens_to_ipc_tokens(ipc_subclass_idx[doc_id], doc_tokens, ipc_subclass_tokens)
            if doc_id in ipc_group_idx:
                add_doc_tokens_to_ipc_tokens(ipc_group_idx[doc_id], doc_tokens, ipc_group_tokens)

            add_doc_tokens_to_doc_index(doc_id, doc_tokens, doc_index)
            add_tokens_to_term_index(doc_id, doc_tokens, term_doc_id_index)

    for ipc_class_name in ipc_class_tokens:
        add_tokens_to_term_index(ipc_class_name, ipc_class_tokens[ipc_class_name], term_ipc_class_idx)
    for ipc_subclass_name in ipc_subclass_tokens:
        add_tokens_to_term_index(ipc_subclass_name, ipc_subclass_tokens[ipc_subclass_name], term_ipc_subclass_idx)
    for ipc_group_name in ipc_group_tokens:
        add_tokens_to_term_index(ipc_group_name, ipc_group_tokens[ipc_group_name], term_ipc_group_idx)

    doc_id_ipc_info = {} # key: doc_id, value: (ipc class, ipc subcass, ipc group)
    for doc_id in ipc_class_idx:
        doc_id_ipc_info[doc_id] = (ipc_class_idx[doc_id], ipc_subclass_idx[doc_id], ipc_group_idx[doc_id])

    ipc_idx = {"ipc_class": term_ipc_class_idx,
                "ipc_subclass": term_ipc_subclass_idx,
                "ipc_group": term_ipc_group_idx,
                "doc_ids": doc_id_ipc_info,
                "ipc_class_count": len(ipc_classes),
                "ipc_subclass_count": len(ipc_subclasses),
                "ipc_group_count": len(ipc_groups)}
    
    return (term_doc_id_index, doc_index, related_documents_index, ipc_idx)

def add_text_tokens_to_doc_tokens(text, doc_tokens):
    if type(text) is str:
        text = text.strip()
    elif type(text) is unicode:
        text = text.encode('ascii', 'replace').strip()
    for word in word_tokenize(text):
        token = stemmer.stem(word).lower()
        if token not in doc_tokens:
            doc_tokens[token] = 0
        doc_tokens[token] += 1

def add_docs_to_related_documents_index(text, doc_id, all_files, related_documents_index):
    text = text.strip()
    doc_ids = set(filter(lambda x: x != "|" and x in all_files, word_tokenize(text)))
    for doc in doc_ids:
        if doc not in all_files: 
            continue
        if doc == doc_id:
            continue
        if doc_id not in related_documents_index:
            related_documents_index[doc_id] = []
        related_documents_index[doc_id].append(doc)

def add_doc_tokens_to_ipc_tokens(ipc, doc_tokens, ipc_tokens):
    if ipc not in ipc_tokens:
        ipc_tokens[ipc] = {}
    for token in doc_tokens:
        if token not in ipc_tokens[ipc]:
            ipc_tokens[ipc][token] = 0
        ipc_tokens[ipc][token] += doc_tokens[token]

def add_doc_tokens_to_term_ipc_index(doc_id, doc_tokens, ipc_idx, term_ipc_idx):
    if doc_id in ipc_idx:
        for token in doc_tokens:
            if token not in term_ipc_idx:
                ipc_idx[token] = {}
            if ipc_idx[doc_id] not in term_ipc_idx[token]:
                term_ipc_idx[token][ipc_idx[doc_id]] = 0
            term_ipc_idx[token][ipc_idx[doc_id]] += doc_tokens[token]

def add_doc_tokens_to_doc_index(doc_id, doc_tokens, doc_index):
    total_document_tokens = sum(doc_tokens.values())
    term_list = map(lambda term: (term, doc_tokens[term]), doc_tokens)
    doc_index[doc_id] = (total_document_tokens, term_list)

def add_tokens_to_term_index(identifier, tokens, term_index):
    log_weighted_tokens = {}
    for token in tokens:
        tf = tokens[token]
        log_weighted_tokens[token] = 1 + log10(tf)
    normalizer = sqrt(reduce(lambda x, y: x + y**2, log_weighted_tokens.values(), 0))
    for token in tokens:
        if token not in term_index:
            term_index[token] = []
        normalized_tf = log_weighted_tokens[token] / normalizer
        term_index[token].append((identifier, tokens[token], normalized_tf))


def write_index(output_dict_file, output_post_file, term_doc_id_index, doc_index, related_documents_index, ipc_idx):
    """
    Writes the index to the output dictionary file and postings file.
    """
    dict_file = file(output_dict_file, "w")
    post_file = file(output_post_file, "w")
    count_bytes = 0

    dict_file.write("## Term ##\n")
    
    count_bytes = write_term_index(dict_file, post_file, term_doc_id_index, len(doc_index), count_bytes)

    count_bytes = write_line_break(post_file, count_bytes)
    dict_file.write("## Doc ID ##\n")
    
    count_bytes = write_doc_index(dict_file, post_file, doc_index, count_bytes)

    count_bytes = write_line_break(post_file, count_bytes)
    dict_file.write("## Fields ##\n")

    count_bytes = write_related_docs_index(dict_file, post_file, related_documents_index, count_bytes)

    dict_file.write("## Doc ID to IPC ##\n")
    write_doc_ipc_info_dict(dict_file, ipc_idx["doc_ids"])

    count_bytes = write_line_break(post_file, count_bytes)
    dict_file.write("## Term to IPC Class ##\n")

    count_bytes = write_term_index(dict_file, post_file, ipc_idx["ipc_class"], ipc_idx["ipc_class_count"], count_bytes)

    count_bytes = write_line_break(post_file, count_bytes)
    dict_file.write("## Term to IPC Subclass ##\n")

    count_bytes = write_term_index(dict_file, post_file, ipc_idx["ipc_subclass"], ipc_idx["ipc_subclass_count"], count_bytes)

    count_bytes = write_line_break(post_file, count_bytes)
    dict_file.write("## Term to IPC Group ##\n")

    count_bytes = write_term_index(dict_file, post_file, ipc_idx["ipc_group"], ipc_idx["ipc_group_count"], count_bytes)

    dict_file.close()
    post_file.close()

def write_term_index(dict_file, post_file, term_index, collection_size, count_bytes):
    for token in term_index:
        postings = term_index[token]
        postings_string = generate_postings_string(postings)
        doc_freq = len(postings)
        idf = log10(float(collection_size) / float(doc_freq))
        dict_string = token + " " + str(count_bytes) + " " + str(idf) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)
    return count_bytes

def write_doc_index(dict_file, post_file, doc_index, count_bytes):
    for doc_id in doc_index:
        num_tokens = doc_index[doc_id][0]
        term_list = doc_index[doc_id][1]
        doc_dict_string = doc_id + " " + str(count_bytes) + " " + str(num_tokens) + "\n"
        dict_file.write(doc_dict_string)
        term_post_string = generate_terms_postings_string(term_list)
        post_file.write(term_post_string)
        count_bytes += len(term_post_string)
    return count_bytes

def write_related_docs_index(dict_file, post_file, related_documents_index, count_bytes):
    for doc_id in related_documents_index:
        fields = related_documents_index[doc_id]
        field_dict_string = doc_id + " " + str(count_bytes) + "\n"
        dict_file.write(field_dict_string)
        field_post_string = generate_fields_postings_string(fields)
        post_file.write(field_post_string)
        count_bytes += len(field_post_string)
    return count_bytes

def write_doc_ipc_info_dict(dict_file, doc_ipc_info):
    for doc_id in doc_ipc_info:
        dict_string = doc_id + " " + doc_ipc_info[doc_id][0] + " " + doc_ipc_info[doc_id][1] + " " + doc_ipc_info[doc_id][2] + "\n"
        dict_file.write(dict_string)

def write_line_break(post_file, count_bytes):
    line_break = "\n"
    post_file.write(line_break)
    count_bytes += len(line_break)
    return count_bytes

def generate_postings_string(postings):
    return reduce(lambda x, y: x + str(y[0]) + " " + str(y[1]) + " " + str(y[2])[2:] + " ", postings, "").strip() + "\n"

def generate_terms_postings_string(term_list):
    return reduce(lambda x, y: x + str(y[0]) + " " + str(y[1]) + " ", term_list, "").strip() + "\n"

def generate_fields_postings_string(fields):
    return reduce(lambda x, y: x + str(y) + " ", fields, "").strip() + "\n"

def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"

document_dir = output_dict_file = output_post_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        document_dir = a
    elif o == '-d':
        output_dict_file = a
    elif o == '-p':
        output_post_file = a
    else:
        assert False, "unhandled option"
if document_dir == None or output_dict_file == None or output_post_file == None:
    usage()
    sys.exit(2)

(term_doc_id_index, doc_index, related_documents_index, ipc_idx) = build_index(document_dir)
write_index(output_dict_file, output_post_file, term_doc_id_index, doc_index, related_documents_index, ipc_idx)