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

def build_index(document_dir):
    """
    Builds the index.
    Returns a tuple with the total number of documents indexed and the index itself.
    """
    term_index = {} # key: terms, value: [(doc_id, tf, norm_tf),...]
    doc_index = {} # key: doc_id, value: (num_tokens, [(term, tf),...])
    family_cites_index = {} # key: doc_id, value: [doc_id,...]

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
                if type(child.text) is not str:
                        continue
                if name in zones:
                    text = child.text.strip()
                    for word in word_tokenize(text):
                        token = stemmer.stem(word).lower()
                        if not is_valid_token(token):
                            continue
                        if token not in doc_tokens:
                            doc_tokens[token] = 0
                        doc_tokens[token] += 1
                elif name in fields:
                    text = child.text.strip()
                    doc_ids = set(filter(lambda x: is_valid_token(x) and x in files_set,word_tokenize(text)))
                    for doc in doc_ids:
                        if not is_valid_token(doc):
                            continue
                        if doc not in files_set:
                            continue
                        if doc_id not in family_cites_index:
                            family_cites_index[doc_id] = []
                        family_cites_index[doc_id].append(doc)

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
                term_index[token].append((doc_id, doc_tokens[token][0], str(normalized_tf)[2:]))

            term_list = map(lambda term: (term, doc_tokens[term][0]), doc_tokens)
            doc_index[doc_id] = (total_document_tokens, term_list)

    return (term_index, doc_index, family_cites_index)

def is_valid_token(token):
    """
    Remove stopwords, punctuation and tokens consisting purely of numbers and punctuation
    """
    is_stop_word = token in stops
    is_puctuation = token in string.punctuation
    is_number_or_punc = reduce(lambda x, y: x and (y in string.digits or y in string.punctuation), token, True)
    return not (is_stop_word or is_puctuation or is_number_or_punc)

def write_index(output_dict_file, output_post_file, term_index, doc_index, family_cites_index):
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

    dict_file.close()
    post_file.close()

def generate_postings_string(postings):
    return reduce(lambda x, y: x + str(y[0]) + " " + str(y[1]) + " " + str(y[2]) + " ", postings, "").strip() + "\n"

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

(term_index, doc_index, family_cites_index) = build_index(document_dir)
write_index(output_dict_file, output_post_file, term_index, doc_index, family_cites_index)