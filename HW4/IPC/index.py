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
zones = ["Title", "Abstract"]
fields = ["Family Members", "Cites"]
ipc_class = ["IPC Class"]
ipc_subclass = ["IPC Subclass"]
ipc_group = ["IPC Group"]

def build_index(document_dir):
    """
    Builds the index.
    Returns a tuple with the total number of documents indexed and the index itself.
    """
    term_index = {} # key: terms, value: [(doc_id, tf, norm_tf),...]
    doc_index = {} # key: doc_id, value: (num_tokens, [(term, tf),...])
    family_cites_index = {} # key: doc_id, value: [doc_id,...]

    ipc_classes = set()
    ipc_subclasses = set()
    ipc_groups = set()

    ipc_class_idx = {} # key: doc_id, value: ipc_class
    ipc_subclass_idx = {} # key: doc_id, value: ipc_subclass
    ipc_group_idx = {} # key: doc_id, value: ipc_group

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
            tree = ET.parse(path)
            root = tree.getroot()
            doc_tokens = {}
            doc_id = f[:-4]

            for child in root:
                name = child.attrib['name'].strip()
                if name in zones:
                    text = ""
                    if type(child.text) is str:
                        text = child.text.strip()
                    elif type(child.text) is unicode:
                        text = child.text.encode('ascii', 'replace')
                    for word in word_tokenize(text):
                        token = stemmer.stem(word).lower()
                        if not is_valid_token(token):
                            continue
                        if token not in doc_tokens:
                            doc_tokens[token] = 0
                        doc_tokens[token] += 1
                elif name in fields:
                    text = child.text.strip()
                    doc_ids = set(filter(lambda x: x != "|" and x in files_set,word_tokenize(text)))
                    for doc in doc_ids:
                        if not is_valid_token(doc):
                            continue
                        if doc not in files_set:
                            continue
                        if doc == doc_id:
                            continue
                        if doc_id not in family_cites_index:
                            family_cites_index[doc_id] = []
                        family_cites_index[doc_id].append(doc)
                elif name in ipc_class:
                    ipc_class_idx[doc_id] = child.text.strip()
                    ipc_classes = ipc_classes | set([child.text.strip()])
                elif name in ipc_subclass:
                    ipc_subclass_idx[doc_id] = child.text.strip()
                    ipc_subclasses = ipc_subclasses | set([child.text.strip()])
                elif name in ipc_group:
                    ipc_group_idx[doc_id] = child.text.strip()
                    ipc_groups = ipc_groups | set([child.text.strip()])

            if doc_id in ipc_class_idx:
                for token in doc_tokens:
                    if token not in term_ipc_class_idx:
                        term_ipc_class_idx[token] = {}
                    if ipc_class_idx[doc_id] not in term_ipc_class_idx[token]:
                        term_ipc_class_idx[token][ipc_class_idx[doc_id]] = 0
                    term_ipc_class_idx[token][ipc_class_idx[doc_id]] += doc_tokens[token]
            if doc_id in ipc_subclass_idx:
                for token in doc_tokens:
                    if token not in term_ipc_subclass_idx:
                        term_ipc_subclass_idx[token] = {}
                    if ipc_subclass_idx[doc_id] not in term_ipc_subclass_idx[token]:
                        term_ipc_subclass_idx[token][ipc_subclass_idx[doc_id]] = 0
                    term_ipc_subclass_idx[token][ipc_subclass_idx[doc_id]] += doc_tokens[token]
            if doc_id in ipc_group_idx:
                for token in doc_tokens:
                    if token not in term_ipc_group_idx:
                        term_ipc_group_idx[token] = {}
                    if ipc_group_idx[doc_id] not in term_ipc_group_idx[token]:
                        term_ipc_group_idx[token][ipc_group_idx[doc_id]] = 0
                    term_ipc_group_idx[token][ipc_group_idx[doc_id]] += doc_tokens[token]

            total_document_tokens = 0
            for token in doc_tokens:
                tf = doc_tokens[token]
                doc_tokens[token] = (tf, 1 + log10(tf))
                total_document_tokens += tf

            normalizer = sqrt(reduce(lambda x, y: x + y**2, map(lambda x: x[1], doc_tokens.values()), 0))

            for token in doc_tokens:
                if token not in term_index:
                    term_index[token] = []
                normalized_tf = doc_tokens[token][1] / normalizer
                term_index[token].append((doc_id, doc_tokens[token][0], normalized_tf))

            term_list = map(lambda term: (term, doc_tokens[term][0]), doc_tokens)
            doc_index[doc_id] = (total_document_tokens, term_list)

    ipc_class_term_idx = {} # key: IPC Class, value: {Term: tf}
    for token in term_ipc_class_idx:
        class_tf_idx = term_ipc_class_idx[token]
        for ipc_class_name in class_tf_idx:
            if ipc_class_name not in ipc_class_term_idx:
                ipc_class_term_idx[ipc_class_name] = {}
            ipc_class_term_idx[ipc_class_name][token] = class_tf_idx[ipc_class_name]
    
    ipc_class_normalizer = {} # key: IPC Class, value: normalizer
    for ipc_class_name in ipc_class_term_idx:
        terms = ipc_class_term_idx[ipc_class_name]
        total = 0
        for term in terms:
            total += (1 + log10(terms[term])) ** 2
        ipc_class_normalizer[ipc_class_name] = sqrt(total)

    for token in term_ipc_class_idx:
        class_tf_idx = term_ipc_class_idx[token]
        for ipc_class_name in class_tf_idx:
            tf = class_tf_idx[ipc_class_name]
            norm_tf = ((1 + log10(tf)) ** 2) / ipc_class_normalizer[ipc_class_name]
            class_tf_idx[ipc_class_name] = (tf, norm_tf)

    for token in term_ipc_class_idx:
        df = len(term_ipc_class_idx[token])
        N = len(ipc_classes)
        idf = log10(float(N) / float(df))
        term_ipc_class_idx[token] = (idf, term_ipc_class_idx[token])

    ipc_subclass_term_idx = {} # key: IPC Subclass, value: {Term: tf}
    for token in term_ipc_subclass_idx:
        subclass_tf_idx = term_ipc_subclass_idx[token]
        for ipc_subclass_name in subclass_tf_idx:
            if ipc_subclass_name not in ipc_subclass_term_idx:
                ipc_subclass_term_idx[ipc_subclass_name] = {}
            ipc_subclass_term_idx[ipc_subclass_name][token] = subclass_tf_idx[ipc_subclass_name]

    ipc_subclass_normalizer = {} # key: IPC Class, value: normalizer
    for ipc_subclass_name in ipc_subclass_term_idx:
        terms = ipc_subclass_term_idx[ipc_subclass_name]
        total = 0
        for term in terms:
            total += (1 + log10(terms[term])) ** 2
        ipc_subclass_normalizer[ipc_subclass_name] = sqrt(total)

    for token in term_ipc_subclass_idx:
        subclass_tf_idx = term_ipc_subclass_idx[token]
        for ipc_subclass_name in subclass_tf_idx:
            tf = subclass_tf_idx[ipc_subclass_name]
            norm_tf = ((1 + log10(tf)) ** 2) / ipc_subclass_normalizer[ipc_subclass_name]
            subclass_tf_idx[ipc_subclass_name] = (tf, norm_tf)

    for token in term_ipc_subclass_idx:
        df = len(term_ipc_subclass_idx[token])
        N = len(ipc_subclasses)
        idf = log10(float(N) / float(df))
        term_ipc_subclass_idx[token] = (idf, term_ipc_subclass_idx[token])

    ipc_group_term_idx = {} # key: IPC Group, value: {Term: tf}
    for token in term_ipc_group_idx:
        group_tf_idx = term_ipc_group_idx[token]
        for ipc_group_name in group_tf_idx:
            if ipc_group_name not in ipc_group_term_idx:
                ipc_group_term_idx[ipc_group_name] = {}
            ipc_group_term_idx[ipc_group_name][token] = group_tf_idx[ipc_group_name]

    ipc_group_normalizer = {} # key: IPC Class, value: normalizer
    for ipc_group_name in ipc_group_term_idx:
        terms = ipc_group_term_idx[ipc_group_name]
        total = 0
        for term in terms:
            total += (1 + log10(terms[term])) ** 2
        ipc_group_normalizer[ipc_group_name] = sqrt(total)

    for token in term_ipc_group_idx:
        group_tf_idx = term_ipc_group_idx[token]
        for ipc_group_name in group_tf_idx:
            tf = group_tf_idx[ipc_group_name]
            norm_tf = ((1 + log10(tf)) ** 2) / ipc_group_normalizer[ipc_group_name]
            group_tf_idx[ipc_group_name] = (tf, norm_tf)

    for token in term_ipc_group_idx:
        df = len(term_ipc_group_idx[token])
        N = len(ipc_groups)
        idf = log10(float(N) / float(df))
        term_ipc_group_idx[token] = (idf, term_ipc_group_idx[token])

    doc_id_ipc_info = {} # key: doc_id, value: (ipc class, ipc subcass, ipc group)
    for doc_id in ipc_class_idx:
        doc_id_ipc_info[doc_id] = (ipc_class_idx[doc_id], ipc_subclass_idx[doc_id], ipc_group_idx[doc_id])

    ipc_idx = {"ipc_class": term_ipc_class_idx,
                "ipc_subclass": term_ipc_subclass_idx,
                "ipc_group": term_ipc_group_idx,
                "doc_ids": doc_id_ipc_info}
    
    return (term_index, doc_index, family_cites_index, ipc_idx)

def is_valid_token(token):
    """
    Remove stopwords, punctuation and tokens consisting purely of numbers and punctuation
    """
    # is_stop_word = token in stops
    # is_puctuation = token in string.punctuation
    # is_number_or_punc = reduce(lambda x, y: x and (y in string.digits or y in string.punctuation), token, True)
    # return not (is_stop_word or is_puctuation or is_number_or_punc)
    return True

def write_index(output_dict_file, output_post_file, term_index, doc_index, family_cites_index, ipc_idx):
    """
    Writes the index to the output dictionary file and postings file.
    The first line of the output dictionary file is the total number of documents indexed.
    """
    dict_file = file(output_dict_file, "w")
    post_file = file(output_post_file, "w")
    count_bytes = 0

    dict_file.write("## Term ##\n")
    for token in term_index:
        postings = term_index[token]
        postings_string = generate_postings_string(postings)
        doc_freq = len(postings)
        idf = log10(float(len(doc_index)) / float(doc_freq))
        dict_string = token + " " + str(count_bytes) + " " + str(idf) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)

    line_break = "\n"
    post_file.write(line_break)
    count_bytes += len(line_break)
    dict_file.write("## Doc ID ##\n")

    for doc_id in doc_index:
        num_tokens = doc_index[doc_id][0]
        term_list = doc_index[doc_id][1]
        doc_dict_string = doc_id + " " + str(count_bytes) + " " + str(num_tokens) + "\n"
        dict_file.write(doc_dict_string)
        term_post_string = generate_terms_postings_string(term_list)
        post_file.write(term_post_string)
        count_bytes += len(term_post_string)

    post_file.write(line_break)
    count_bytes += len(line_break)
    dict_file.write("## Fields ##\n")

    for doc_id in family_cites_index:
        fields = family_cites_index[doc_id]
        field_dict_string = doc_id + " " + str(count_bytes) + "\n"
        dict_file.write(field_dict_string)
        field_post_string = generate_fields_postings_string(fields)
        post_file.write(field_post_string)
        count_bytes += len(field_post_string)

    dict_file.write("## Doc ID to IPC ##\n")
    for doc_id in ipc_idx["doc_ids"]:
        dict_string = doc_id + " " + ipc_idx["doc_ids"][doc_id][0] + " " + ipc_idx["doc_ids"][doc_id][1] + " " + ipc_idx["doc_ids"][doc_id][2] + "\n"
        dict_file.write(dict_string)

    post_file.write(line_break)
    count_bytes += len(line_break)
    dict_file.write("## Term to IPC Class ##\n")

    term_index = ipc_idx["ipc_class"]
    for token in term_index:
        postings = term_index[token][1]
        postings_list = map(lambda x: (x, postings[x][0], postings[x][1]), postings)
        postings_string = generate_postings_string(postings_list)
        idf = term_index[token][0]
        dict_string = token + " " + str(count_bytes) + " " + str(idf) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)

    post_file.write(line_break)
    count_bytes += len(line_break)
    dict_file.write("## Term to IPC Subclass ##\n")

    term_index = ipc_idx["ipc_subclass"]
    for token in term_index:
        postings = term_index[token][1]
        postings_list = map(lambda x: (x, postings[x][0], postings[x][1]), postings)
        postings_string = generate_postings_string(postings_list)
        idf = term_index[token][0]
        dict_string = token + " " + str(count_bytes) + " " + str(idf) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)

    post_file.write(line_break)
    count_bytes += len(line_break)
    dict_file.write("## Term to IPC Group ##\n")

    term_index = ipc_idx["ipc_group"]
    for token in term_index:
        postings = term_index[token][1]
        postings_list = map(lambda x: (x, postings[x][0], postings[x][1]), postings)
        postings_string = generate_postings_string(postings_list)
        idf = term_index[token][0]
        dict_string = token + " " + str(count_bytes) + " " + str(idf) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)

    dict_file.close()
    post_file.close()

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

(term_index, doc_index, family_cites_index, ipc_idx) = build_index(document_dir)
write_index(output_dict_file, output_post_file, term_index, doc_index, family_cites_index, ipc_idx)