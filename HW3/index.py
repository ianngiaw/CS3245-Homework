#!/usr/bin/python
import re
import nltk
import sys
import getopt

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize, word_tokenize

stemmer = nltk.stem.porter.PorterStemmer()

def build_index(document_dir):
    """
    Builds the index
    """
    index = {}
    files = listdir(document_dir)
    files.remove(".DS_Store")
    files.sort(key=lambda f: int(f))
    for f in files:
        path = join(document_dir, f)
        if isfile(path):
            input_file = file(path, 'r')
            for line in input_file:
                for sent in sent_tokenize(line):
                    for word in word_tokenize(sent):
                        stemmed_word = stemmer.stem(word)
                        token = stemmed_word.lower()
                        if token not in index:
                            index[token] = []
                        if len(index[token]) == 0 or index[token][-1][0] != f:
                            index[token].append([f, 1])
                        else:
                            index[token][-1][1] += 1
    return index

def write_index(output_dict_file, output_post_file, index):
    """
    Writes the index to the output dictionary file and postings file
    """
    dict_file = file(output_dict_file, "w")
    post_file = file(output_post_file, "w")
    count_bytes = 0
    for token in index:
        postings = index[token]
        postings_string = generate_postings_string(postings)
        dict_string = token + " " + str(count_bytes) + " " + str(len(postings)) + "\n"
        dict_file.write(dict_string)
        post_file.write(postings_string)
        count_bytes += len(postings_string)
    dict_file.close()
    post_file.close()

def generate_postings_string(postings):
    """
    Generates a string that is written to a postings file. The string is formatted as follows:
    "doc_id1 term_freq1 doc_id2 term_freq2 doc_id3 term_freq3...doc_idx term_freqx\n"
    """
    return reduce(lambda x, y: x + str(y[0]) + " " + str(y[1]) + " ", postings, "").strip() + "\n"

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

index = build_index(document_dir)
write_index(output_dict_file, output_post_file, index)